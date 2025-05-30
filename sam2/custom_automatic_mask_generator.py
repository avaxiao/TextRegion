from .automatic_mask_generator import *
import torch
from sam2.custom_sam2_image_predictor import CustomSAM2ImagePredictor
from sam2.utils.custom_amg import CustomMaskData



def box_iou(boxes1, boxes2):
    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0)

    x1A = b1[..., 0]
    y1A = b1[..., 1]
    x2A = b1[..., 2]
    y2A = b1[..., 3]

    x1B = b2[..., 0]
    y1B = b2[..., 1]
    x2B = b2[..., 2]
    y2B = b2[..., 3]

    inter_xmin = torch.max(x1A, x1B)
    inter_ymin = torch.max(y1A, y1B)
    inter_xmax = torch.min(x2A, x2B)
    inter_ymax = torch.min(y2A, y2B)

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    areaA = (x2A - x1A).clamp(min=0) * (y2A - y1A).clamp(min=0)
    areaB = (x2B - x1B).clamp(min=0) * (y2B - y1B).clamp(min=0)

    union = areaA + areaB - inter_area

    iou_matrix = inter_area / (union + 1e-8)
    return iou_matrix


def mask_containment(maskA, maskB):
    areaA = maskA.sum()
    areaB = maskB.sum()

    intersection = torch.minimum(maskA, maskB).sum()
    smaller_area = torch.minimum(areaA, areaB)

    if smaller_area < 1e-8:
        return torch.tensor(0.0, device=maskA.device, dtype=maskA.dtype)

    return intersection / smaller_area


def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh[..., 0], box_xywh[..., 1], box_xywh[..., 2], box_xywh[..., 3]
    x2 = x + w
    y2 = y + h
    return torch.stack((x, y, x2, y2), dim=-1)


class CustomAutomaticMaskGenerator(SAM2AutomaticMaskGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.predictor = CustomSAM2ImagePredictor(
            sam_model=kwargs.get('model'),
            max_hole_area=kwargs.get('min_mask_region_area', 0),
            max_sprinkle_area=kwargs.get('min_mask_region_area', 0),
        )

        self.min_size = kwargs.get('min_size', 100)
        self.return_non_overlapping = kwargs.get('return_non_overlapping', False)
        if not kwargs.get('multimask_output', False):
            self.return_non_overlapping = False

        self.fuse_mask = kwargs.get('fuse_mask', True)
        self.fuse_mask_threshold = kwargs.get('fuse_mask_threshold', 0.8)

        if self.fuse_mask or self.return_non_overlapping:
            self.split_mask_by_size = True
        else:
            self.split_mask_by_size = False

        self.return_bool = kwargs.get('return_bool', False)


    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2AutomaticMaskGenerator): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        kwargs["model"] = sam_model
        return cls(**kwargs)


    @torch.no_grad()
    def fuse_overlap_masks(self, ranking_masks_by_prediou_list, key_name="segmentations"):
        fuse_mask_list = []
        ranking_masks_tensor = torch.stack(
            [mask[key_name] for mask in ranking_masks_by_prediou_list]).float()
        sums = ranking_masks_tensor.abs().sum(dim=(1, 2))  # shape = [N]
        mask_non_zero = (sums > 0)
        masks_filter_none = ranking_masks_tensor[mask_non_zero]

        recording = torch.ones(len(masks_filter_none), dtype=torch.bool, device=masks_filter_none.device)
        bboxes_first = torch.tensor([d["bbox"] for d in ranking_masks_by_prediou_list], device=masks_filter_none.device)
        bboxes = bboxes_first[mask_non_zero]

        bboxes = xywh_to_xyxy(bboxes)
        bbox_iou_matrix = box_iou(bboxes, bboxes)  # (N,N)

        for i in range(len(masks_filter_none)):
            if not recording[i]:
                continue

            recording[i] = False

            current_mask = masks_filter_none[i]
            candidate_indices = (bbox_iou_matrix[i] > 0.2) & recording

            group_indices = [i]

            if candidate_indices.any():
                idxs = candidate_indices.nonzero().flatten()
                same_group = []
                for j in idxs:
                    iou_val = mask_containment(current_mask, masks_filter_none[j])
                    if iou_val > self.fuse_mask_threshold:
                        same_group.append(j.item())

                if same_group:
                    group_indices.extend(same_group)

                group_indices = torch.tensor(group_indices, device=masks_filter_none.device)
                recording[group_indices] = False

                group_masks = masks_filter_none[group_indices]
                merged_mask = group_masks.amax(dim=0)
                fuse_mask_list.append({key_name: merged_mask})

            else:
                fuse_mask_list.append({key_name: current_mask})
        return fuse_mask_list

    @torch.no_grad()
    def generate(self, image: Optional[np.ndarray] = None, image_tensor_for_sam2: Optional[torch.Tensor] = None,
                 size_hw: Optional[list] = None) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        assert self.output_mode == "binary_mask"
        # Generate masks
        mask_data = self._generate_masks(image, image_tensor_for_sam2, size_hw)
        # Encode masks
        # if self.output_mode == "coco_rle":
        #     mask_data["segmentations"] = [
        #         coco_encode_rle(rle) for rle in mask_data["rles"]
        #     ]
        # elif self.output_mode == "binary_mask":
        #     mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        # else:
        #     mask_data["segmentations"] = mask_data["rles"]

        if image is not None:
            orig_size = image.shape[:2]
        else:
            orig_size = size_hw

        mask_data["mask_size"] = (mask_data["masks"] > self.mask_threshold).sum(dim=(1, 2))
        if self.return_bool:
            mask_data["masks"] = mask_data["masks"] > self.mask_threshold
        else:
            mask_data["masks"] = torch.clamp(mask_data["masks"], min=0, max=3)

        mask_data["masks"] = self.predictor._transforms.postprocess_masks(
            mask_data["masks"].unsqueeze(1), orig_size
        )[:, 0].half()



        # Write mask records
        curr_anns = []

        if self.return_non_overlapping or self.fuse_mask:
            sub_parts_masks = []
            parts_masks = []
            whole_masks = []

        for idx in range(len(mask_data["masks"])):
            ann = {
                "segmentation": mask_data["masks"][idx],
                # "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                # "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }

            if mask_data["mask_size"][idx] <= self.min_size:
                continue

            if self.return_non_overlapping or self.fuse_mask:
                if mask_data["sizes"][idx] == 2:
                    whole_masks.append(ann)
                elif mask_data["sizes"][idx] == 1:
                    parts_masks.append(ann)
                elif mask_data["sizes"][idx] == 0:
                    sub_parts_masks.append(ann)
            else:
                curr_anns.append(ann)

        if self.return_non_overlapping or self.fuse_mask:
            non_overlapping_mask = torch.full(orig_size, -1, dtype=int, device=mask_data["masks"].device)
            whole_masks = sorted(whole_masks, key=(lambda x: x['predicted_iou']), reverse=True)
            parts_masks = sorted(parts_masks, key=(lambda x: x['predicted_iou']), reverse=True)
            sub_parts_masks = sorted(sub_parts_masks, key=(lambda x: x['predicted_iou']), reverse=True)
            curr_anns = whole_masks + parts_masks + sub_parts_masks

        if self.fuse_mask:
            curr_anns = self.fuse_overlap_masks(curr_anns, "segmentation")

        non_overlapping_list = []
        if self.return_non_overlapping:
            for ann in curr_anns:
                if (non_overlapping_mask == -1).sum() < self.min_size:
                    break
                non_mask_area = (non_overlapping_mask == -1)
                to_mask_area_single = (ann['segmentation'] > 0) & non_mask_area
                if to_mask_area_single.sum() < self.min_size:
                    continue
                ann['segmentation'] = ann['segmentation'] * to_mask_area_single
                non_overlapping_mask[to_mask_area_single] = 1
                non_overlapping_list.append(ann)
            curr_anns = non_overlapping_list

        return curr_anns

    @torch.no_grad()
    def _generate_masks(self, image: Optional[np.ndarray] = None, image_tensor_for_sam2: Optional[torch.Tensor] = None,
                        size_hw: Optional[list] = None) -> CustomMaskData:

        # Iterate over image crops
        data = CustomMaskData()

        if image is not None:
            orig_size = image.shape[:2]
        else:
            orig_size = size_hw

        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        if image_tensor_for_sam2 is not None:
            assert len(layer_idxs) == 1

        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size, image_tensor_for_sam2, size_hw)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        # data.to_numpy()

        return data

    @torch.no_grad()
    def _process_crop(
            self,
            image: Optional[np.ndarray] = None,
            crop_box: Optional[List[int]] = None,
            crop_layer_idx: Optional[int] = None,
            orig_size: Optional[Tuple[int, ...]] = None,
            image_tensor_for_sam2: Optional[torch.Tensor] = None,
            size_hw: Optional[list] = None
    ) -> CustomMaskData:
        # Crop the image and calculate embeddings

        if image_tensor_for_sam2 is None:
            x0, y0, x1, y1 = crop_box
            cropped_im = image[y0:y1, x0:x1, :]
            cropped_im_size = cropped_im.shape[:2]
            self.predictor.set_image(cropped_im, image_tensor_for_sam2)
        else:
            if size_hw is None:
                cropped_im_size = image.shape[:2]
            else:
                cropped_im_size = size_hw
            self.predictor.set_image(None, image_tensor_for_sam2, size_hw)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = CustomMaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch_for_single_image(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_predictor()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        # data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        return data

    def _process_batch_for_single_image(
            self,
            points: np.ndarray,
            im_size: Tuple[int, ...],
            crop_box: List[int],
            orig_size: Tuple[int, ...],
            normalize=False,
            img_idx: int = -1,
    ) -> CustomMaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        low_res_masks, iou_preds = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True
        )

        # Serialize predictions and store in CustomMaskData
        if self.split_mask_by_size:
            low_res_masks = low_res_masks.half()
            sub_parts_masks = low_res_masks[:, 0, :, :]
            parts_masks = low_res_masks[:, 1, :, :]
            whole_masks = low_res_masks[:, 2, :, :]
            low_res_masks = torch.concat([whole_masks, parts_masks, sub_parts_masks], dim=0)

            sub_parts_iou_preds = iou_preds[:, 0]
            parts_iou_preds = iou_preds[:, 1]
            whole_iou_preds = iou_preds[:, 2]
            iou_preds = torch.stack([whole_iou_preds, parts_iou_preds, sub_parts_iou_preds], dim=-1)

            sizes = torch.tensor([2] * len(whole_masks) + [1] * len(parts_masks) + [0] * len(sub_parts_masks))

            data = CustomMaskData(
                iou_preds=iou_preds.flatten(0, 1),
                points=torch.concat([points] * 3),
                masks=low_res_masks,
                sizes=sizes.to(self.predictor.device),
            )
        else:
            data = CustomMaskData(
                # masks=masks.flatten(0, 1).half(),
                iou_preds=iou_preds.flatten(0, 1),
                points=points.repeat_interleave(low_res_masks.shape[1], dim=0),
                masks=low_res_masks.flatten(0, 1).half(),
            )

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = iou_preds.flatten(0, 1) > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            raise NotImplementedError

        # Threshold masks and calculate boxes
        # data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"] > self.mask_threshold)

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        # data["rles"] = mask_to_rle_pytorch(data["masks"])
        # del data["masks"]

        return data

    @torch.no_grad()
    def _generate_masks_for_batch(self) -> List:

        mask_data_list = []
        for img_idx in range(len(self.predictor._orig_hw)):
            # Iterate over image crops
            data = CustomMaskData()
            orig_size = self.predictor._orig_hw[img_idx]

            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, self.crop_n_layers, self.crop_overlap_ratio
            )
            assert len(layer_idxs) == 1

            for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
                crop_data = self._process_crop_for_batch(img_idx, crop_box, layer_idx, orig_size)
                data.cat(crop_data)

            # Remove duplicate masks between crops
            if len(crop_boxes) > 1:
                # Prefer masks from smaller crops
                scores = 1 / box_area(data["crop_boxes"])
                scores = scores.to(data["boxes"].device)
                keep_by_nms = batched_nms(
                    data["boxes"].float(),
                    scores,
                    torch.zeros_like(data["boxes"][:, 0]),  # categories
                    iou_threshold=self.crop_nms_thresh,
                )
                data.filter(keep_by_nms)
            mask_data_list.append(data)

        self.predictor.reset_predictor()

        return mask_data_list

    @torch.no_grad()
    def generate_for_batch(self, image_tensor_for_sam2: Optional[torch.Tensor] = None,
                           size_hw_list: Optional[list] = None,
                           image: Optional[list] = None) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        assert self.output_mode == "binary_mask"
        # Generate masks

        self.predictor.set_image_batch(image_tensor_for_sam2, size_hw_list, image)
        mask_data_list = self._generate_masks_for_batch()

        if size_hw_list is not None:
            _orig_hw = size_hw_list
        else:
            _orig_hw = [img.shape[:2] for img in image]

        # Write mask records
        all_anns = []
        for img_idx, mask_data in enumerate(mask_data_list):
            curr_anns = []

            mask_data["mask_size"] = (mask_data["masks"] > self.mask_threshold).sum(dim=(1, 2))
            if self.return_bool:
                mask_data["masks"] = mask_data["masks"] > self.mask_threshold
            else:
                mask_data["masks"] = torch.clamp(mask_data["masks"], min=0, max=3)


            mask_data["masks"] = self.predictor._transforms.postprocess_masks(
                mask_data["masks"].unsqueeze(1), _orig_hw[img_idx]
            )[:, 0].half()


            if self.return_non_overlapping or self.fuse_mask:
                sub_parts_masks = []
                parts_masks = []
                whole_masks = []

            for idx in range(len(mask_data["masks"])):

                if mask_data["mask_size"][idx] <= self.min_size:
                    continue

                ann = {
                    "segmentations": mask_data["masks"][idx],
                    # "area": area_from_rle(mask_data["rles"][idx]),
                    "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                    "predicted_iou": mask_data["iou_preds"][idx].item(),
                    "point_coords": [mask_data["points"][idx].tolist()],
                    "stability_score": mask_data["stability_score"][idx].item(),
                }

                if self.return_non_overlapping or self.fuse_mask:
                    if mask_data["sizes"][idx] == 2:
                        whole_masks.append(ann)
                    elif mask_data["sizes"][idx] == 1:
                        parts_masks.append(ann)
                    elif mask_data["sizes"][idx] == 0:
                        sub_parts_masks.append(ann)
                else:
                    curr_anns.append(ann)

            if self.return_non_overlapping or self.fuse_mask:
                non_overlapping_mask = torch.full(_orig_hw[img_idx], -1, dtype=int, device=mask_data["masks"].device)
                whole_masks = sorted(whole_masks, key=(lambda x: x['predicted_iou']), reverse=True)
                parts_masks = sorted(parts_masks, key=(lambda x: x['predicted_iou']), reverse=True)
                sub_parts_masks = sorted(sub_parts_masks, key=(lambda x: x['predicted_iou']), reverse=True)
                curr_anns = whole_masks + parts_masks + sub_parts_masks

            if self.fuse_mask:
                curr_anns = self.fuse_overlap_masks(curr_anns, "segmentations")

            non_overlapping_list = []
            if self.return_non_overlapping:
                for ann in curr_anns:
                    if (non_overlapping_mask == -1).sum() < self.min_size:
                        break
                    non_mask_area = (non_overlapping_mask == -1)
                    to_mask_area_single = (ann['segmentations'] > 0) & non_mask_area
                    if to_mask_area_single.sum() < self.min_size:
                        continue
                    ann['segmentations'] = ann['segmentations'] * to_mask_area_single
                    non_overlapping_mask[to_mask_area_single] = 1
                    non_overlapping_list.append(ann)
                curr_anns = non_overlapping_list

            all_anns.append(curr_anns)
        return all_anns

    @torch.no_grad()
    def _process_crop_for_batch(
            self,
            img_idx: int,
            crop_box: Optional[List[int]] = None,
            crop_layer_idx: Optional[int] = None,
            orig_size: Optional[Tuple[int, ...]] = None,
    ) -> CustomMaskData:
        # Crop the image and calculate embeddings

        # Get points for this crop
        points_scale = np.array(orig_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = CustomMaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch_for_batch(
                points, orig_size, crop_box, orig_size, normalize=True, img_idx=img_idx
            )
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        # data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        return data

    def _process_batch_for_batch(
            self,
            points: np.ndarray,
            im_size: Tuple[int, ...],
            crop_box: List[int],
            orig_size: Tuple[int, ...],
            normalize=False,
            img_idx: int = -1,
    ) -> CustomMaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        low_res_masks, iou_preds = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
            img_idx=img_idx
        )

        # Serialize predictions and store in CustomMaskData
        if self.split_mask_by_size:
            low_res_masks = low_res_masks.half()
            sub_parts_masks = low_res_masks[:, 0, :, :]
            parts_masks = low_res_masks[:, 1, :, :]
            whole_masks = low_res_masks[:, 2, :, :]
            low_res_masks = torch.concat([whole_masks, parts_masks, sub_parts_masks], dim=0)

            sub_parts_iou_preds = iou_preds[:, 0]
            parts_iou_preds = iou_preds[:, 1]
            whole_iou_preds = iou_preds[:, 2]
            iou_preds = torch.concat([whole_iou_preds, parts_iou_preds, sub_parts_iou_preds], dim=0)

            sizes = torch.tensor([2] * len(whole_masks) + [1] * len(parts_masks) + [0] * len(sub_parts_masks))

            data = CustomMaskData(
                iou_preds=iou_preds,
                points=torch.concat([points] * 3),
                masks=low_res_masks,
                sizes=sizes.to(self.predictor.device),
            )
        else:
            data = CustomMaskData(
                # masks=masks.flatten(0, 1).half(),
                iou_preds=iou_preds.flatten(0, 1),
                points=points.repeat_interleave(low_res_masks.shape[1], dim=0),
                masks=low_res_masks.flatten(0, 1).half(),
            )

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            # One step refinement using previous mask predictions
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Threshold masks and calculate boxes
        # data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"] > self.mask_threshold)

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        # data["rles"] = mask_to_rle_pytorch(data["masks"])
        # del data["masks"]
        return data
