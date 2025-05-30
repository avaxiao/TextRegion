import os
import pytz
from datetime import datetime

from utils.visualize_segmentation import *

import numpy as np
from collections import defaultdict
from typing import Dict, Any, Callable, List, Tuple, NamedTuple, Text, Optional
from TextRegionSegmenter import add_segment_config, CLIPWithSAM2ForSegmentation
from tqdm import tqdm
import json
import argparse
import torch
import torch.nn.functional as F



def mask_to_bbox(binary_mask):
    """
    Calculate the bounding box for a binary mask.

    Args:
        binary_mask (np.ndarray): A 2D binary mask (values 0 and 1).

    Returns:
        np.ndarray: Bounding box in xyxy format (x_min, y_min, x_max, y_max).
    """
    # Find indices where the mask is 1
    coords = np.argwhere(binary_mask > 0)

    # If no positive values are found, return None or an empty array
    if coords.size == 0:
        return np.array([])

    # coords are in (y, x) format
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    return np.array([x_min, y_min, x_max, y_max])


def iou(box1, box2):
    x1 = max(box1.x, box2.x)
    x2 = max(x1, min(box1.x+box1.w, box2.x+box2.w))
    y1 = max(box1.y, box2.y)
    y2 = max(y1, min(box1.y+box1.h, box2.y+box2.h))
    intersection = Box(x=x1, y=y1, w=x2-x1, h=y2-y1)
    intersection_area = intersection.area
    union_area = box1.area+box2.area-intersection_area
    return intersection_area / union_area


class Box(NamedTuple):
    x: int
    y: int
    w: int = 0
    h: int = 0

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center(self):
        return Box(self.x + self.w // 2, self.y + self.h // 2)

    def corners(self):
        yield Box(self.x, self.y)
        yield Box(self.x + self.w, self.y)
        yield Box(self.x + self.w, self.y + self.h)
        yield Box(self.x, self.y + self.h)

    @property
    def area(self):
        return self.w * self.h

    def intersect(self, other: "Box") -> "Box":
        x1 = max(self.x, other.x)
        x2 = max(x1, min(self.x+self.w, other.x+other.w))
        y1 = max(self.y, other.y)
        y2 = max(y1, min(self.y+self.h, other.y+other.h))
        return Box(x=x1, y=y1, w=x2-x1, h=y2-y1)

    def min_bounding(self, other: "Box") -> "Box":
        corners = list(self.corners())
        corners.extend(other.corners())
        min_x = min_y = float("inf")
        max_x = max_y = -float("inf")

        for item in corners:
            min_x = min(min_x, item.x)
            min_y = min(min_y, item.y)
            max_x = max(max_x, item.x)
            max_y = max(max_y, item.y)

        return Box(min_x, min_y, max_x - min_x, max_y - min_y)


    def expand(self, growth: float = .1) -> "Box":
        factor = 1 + growth
        w = factor * self.w
        h = factor * self.h
        return Box(min_x - (w - self.w) / 2, min_y - (h - self.h) / 2, w, h)


def clean_mask(mask, min_area=100, keep_largest=False):
    if keep_largest:
        # Find connected components
        num_labels, labels_im = cv2.connectedComponents(mask)
        # Find the largest component (ignoring background label 0)
        max_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
        # Create mask with only the largest component
        refined_mask = np.uint8(labels_im == max_label)
    else:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        refined_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined_mask[labels == i] = 1
    return refined_mask


def eval_benchmark(args, segment_args, clip_sam2_segmenter, input_file):

    input_file_dir = os.path.join(args.input_file_root, input_file + '.jsonl')
    with open(input_file_dir) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    correct_count = 0
    total_count = 0

    detector_file_dir = input_file.split('_')[0] + '_dets_dict.json'
    detector_file_dir = os.path.join(args.input_file_root, detector_file_dir)

    detector_file = open(detector_file_dir)
    detections_list = json.load(detector_file)
    if isinstance(detections_list, dict):
        detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
    else:
        detections_map = defaultdict(list)
        for detection in detections_list:
            detections_map[detection["image_id"]].append(detection["box"])


    for datum in tqdm(data):
        if "coco" in datum["file_name"].lower():
            file_name = "_".join(datum["file_name"].split("_")[:-1]) + ".jpg"
        else:
            file_name = datum["file_name"]
        img_dir = os.path.join(args.image_root, file_name)
        args.image_dir = segment_args.image_dir = img_dir

        gold_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in datum["anns"]]
        if isinstance(datum["ann_id"], int) or isinstance(datum["ann_id"], str):
            datum["ann_id"] = [datum["ann_id"]]
        assert isinstance(datum["ann_id"], list)
        gold_index = [i for i in range(len(datum["anns"])) if datum["anns"][i]["id"] in datum["ann_id"]]

        image = Image.open(args.image_dir).convert("RGB")
        image_size = image.size[::-1]

        region_features, unique_masks = clip_sam2_segmenter.predict(segment_args, compute_logits=False)
        unique_masks = F.interpolate(unique_masks.unsqueeze(0), size=image_size, mode='bilinear')[0]
        filter_masks = unique_masks


        box_detect = [Box(x=box[0], y=box[1], w=box[2], h=box[3]) for box in detections_map[int(datum["image_id"])]]

        for sentence in datum["sentences"]:
            query_words = [sentence["raw"].lower()]
            _ = clip_sam2_segmenter.encode_query_words(query_words)
            region_logits = region_features[0] @ clip_sam2_segmenter.query_features.T
            seg_preds = filter_masks[region_logits.argmax(0)]

            binary_mask = (seg_preds > 0).int()[0].cpu().numpy().astype(np.uint8)
            binary_mask = clean_mask(binary_mask, keep_largest=True)

            x_min, y_min, x_max, y_max = mask_to_bbox(binary_mask)
            box_pred = Box(x=x_min, y=y_min, w=x_max-x_min , h=y_max-y_min)

            biggest_iou = 0
            box_choose = None
            for box in box_detect:
                iou_new = iou(box, box_pred)
                if iou_new > biggest_iou:
                    biggest_iou = iou_new
                    box_choose = box
            if box_choose is not None:
                box_pred = box_choose


            correct = False
            for g_index in gold_index:
                try:
                    if iou(box_pred, gold_boxes[g_index]) > 0.5:
                        correct = True
                        break
                except:
                    pass


            if correct:
                correct_count += 1

            total_count += 1
            print(f"est_acc: {100 * correct_count / total_count:.3f}")
    return correct_count, total_count



def main(args, segment_args):

    eval_list = [
        "refcoco_val",
        "refcoco_testa",
        "refcoco_testb",
        "refcoco+_val",
        "refcoco+_testa",
        "refcoco+_testb",
        "refcocog_val",
        "refcocog_test",
    ]

    clip_sam2_segmenter = CLIPWithSAM2ForSegmentation(segment_args)

    if args.record_log:
        chicagoTz = pytz.timezone("America/Chicago")
        timeInChicago = datetime.now(chicagoTz)
        flag = timeInChicago.strftime("%m_%d_%Y_%H_%M")

        save_log_dir = os.path.join(args.log_dir, args.clip_pretrained + '_' + flag + '.txt')
        os.makedirs(args.log_dir, exist_ok=True)

        with open(save_log_dir, 'w') as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n\n")
            for key, value in vars(segment_args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n\n")


    for input_file in eval_list:
        correct_count, total_count = eval_benchmark(args, segment_args, clip_sam2_segmenter, input_file)
        print(f"{input_file}: acc: {100 * correct_count / total_count:.3f}, correct: {correct_count}, total: {total_count}")

        if args.record_log:
            with open(save_log_dir, 'a+') as f:
                f.write(f"{input_file}: acc: {100 * correct_count / total_count:.3f}, correct: {correct_count}, total: {total_count}")
                f.write(f"\n\n")




def parse_args(cl_args: List[str] = None):
    parser = argparse.ArgumentParser(description="Evaluate ReferralSeg")

    parser.add_argument("--input_file_root", default='/shared/nas2/yaox11/data/datasets/coco_rec/reclip_data')
    parser.add_argument("--image_root", default='/shared/nas2/yaox11/data/datasets/coco_rec/train2014')
    parser.add_argument("--log_dir", default="./eval_results/referring")

    parser.add_argument("--clip_architecture", type=str, default='ViT-L/14@336px',
                        choices=['ViT-L/14@336px', 'PE-Core-L14-336', 'ViT-L-16-SigLIP2-256'])

    parser.add_argument("--scale", nargs='+', default=[2016, 2016])
    parser.add_argument("--record_log", type=eval, choices=[True, False], default=True)
    return parser.parse_args(cl_args)



if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    if args.clip_architecture == 'ViT-L/14@336px':
        args.clip_pretrained = 'openai'
    elif args.clip_architecture == 'PE-Core-L14-336':
        args.clip_pretrained = 'meta'
    elif args.clip_architecture == 'ViT-L-16-SigLIP2-256':
        args.clip_pretrained = 'siglip2'
    else:
        raise ValueError(f"Unknown clip_architecture: {args.clip_architecture}")


    cl_args = []
    for k, v in args.__dict__.items():
        arg_name = f'--{k}'
        if arg_name not in cl_args:
            cl_args.extend([arg_name, str(v)])

    segment_args = add_segment_config(cl_args)
    segment_args.scale = eval(segment_args.scale[0])
    args.precision = segment_args.dtype

    with torch.inference_mode(), torch.autocast("cuda", dtype=args.precision):
        main(args, segment_args)
