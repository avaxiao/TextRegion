from TextRegionSegmenter import add_segment_config, load_sam2
import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = f"23333"

from functools import partial
from utils.LISA.dataset import ValDataset, collate_fn
from utils.LISA.utils import AverageMeter, Summary, dict_to_cuda, intersectionAndUnionGPU
import torch.distributed as dist
import json
import os.path
import custom_clip
from utils.imagenet_template import openai_imagenet_template
from torchvision.transforms import v2
from custom_clip.clip import tokenize
from utils.visualize_segmentation import *
import torch
import numpy as np
from custom_open_clip import create_model, tokenizer
import argparse
from typing import List
from mmcv.image import imrescale
from tqdm import tqdm
import pytz
from datetime import datetime
import torch.nn.functional as F



dist.init_process_group(backend="nccl", init_method="env://")



class TextRegionReasonSegmenter():
    def __init__(self, args, region_logit_scale=50, prob_thd=0.0,
                 device=torch.device("cuda") if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.dtype = args.dtype
        self.resize_method = args.resize_method
        self.encode_global_image = True
        self.crop_size = args.crop_size
        self.region_logit_scale = region_logit_scale
        self.prob_thd = prob_thd

        if args.global_patch_threshold is not None:
            self.global_patch_threshold = args.global_patch_threshold
        else:
            self.global_patch_threshold = 0.07

        clip_preprocess = v2.Compose(
            [
                v2.Resize((args.crop_size, args.crop_size), interpolation=Image.BICUBIC),
                v2.ToDtype(self.dtype, scale=True),
                v2.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.clip_pretrained = args.clip_pretrained
        if args.clip_pretrained == 'openai':
            self.clip, _ = custom_clip.load(args.clip_architecture, device=device, jit=False, download_root=args.clip_download_root)
            self.patch_size = self.clip.visual.patch_size
        else:
            self.clip = create_model(args.clip_architecture, pretrained=args.clip_pretrained, precision=args.dtype, cache_dir=args.clip_download_root)
            self.clip.eval().to(device)
            self.patch_size = self.clip.visual.patch_size[0]

        self.clip_preprocess = clip_preprocess

        self.points_per_side = args.points_per_side
        self.sam2_generator = load_sam2(args, args.model_cfg, args.sam2_checkpoint)
        self.sam2_generator.min_size = 0
        self.sam_transform = self.sam2_generator.predictor._transforms


    def encode_query_words(self, query_words):
        if type(query_words) == str:
            query_words = [query_words]

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                qw_list = tokenize([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                feature = self.clip.encode_text(qw_list)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        query_features = torch.cat(query_features, dim=0)
        self.query_features = query_features.to(self.device, dtype=self.dtype)
        return self.query_features


    def clip_value_with_sam2_attn(self, args, unique_low_res_masks, clip_v, blk):
        attn_layer = blk.attn
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = clip_v.size()
        head_dim = embed_dim // num_heads

        x = blk.ln_1(clip_v)
        q, k, v_ori = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)

        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            v = v_ori[1:, :, :].permute(1, 2, 0).contiguous().view(bsz, embed_dim, patch_num, patch_num)

            if self.encode_global_image:
                crop_id = 1
                v_multi_reso = F.interpolate(v[:1], [self.points_per_h, self.points_per_w], mode="bilinear")
            else:
                crop_id = 0
                v_multi_reso = torch.zeros(1, embed_dim, self.points_per_h, self.points_per_w, device=self.device, dtype=self.dtype)

            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num

                    if self.encode_global_image :
                        v_multi_reso[:, :, y1:y2, x1:x2] = v_multi_reso[:, :, y1:y2, x1:x2] + v[crop_id]
                    else:
                        v_multi_reso[:, :, y1:y2, x1:x2] = v[crop_id]
                    crop_id += 1

            bsz = 1
            v_single_head = v_multi_reso.contiguous().view(1, embed_dim, self.points_per_h * self.points_per_w)
            v_multi_head = v_single_head.contiguous().view(bsz * num_heads, head_dim, self.points_per_h * self.points_per_w).permute(0, 2, 1)

        else:
            raise NotImplementedError

        if args.remove_global_patch:
            patch_features = v_single_head.permute(0, 2, 1)[0]
            patch_features /= patch_features.norm(dim=-1, keepdim=True)
            patch_similarity = patch_features @ patch_features.T

            patch_2_region = patch_similarity @ (unique_low_res_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (unique_low_res_masks > 0).sum(dim=-1)

            belong_score = patch_2_region_avg * (unique_low_res_masks > 0).float().T
            belong_score_avg = belong_score.sum(dim=-1) / ((unique_low_res_masks > 0).sum(dim=0) + 1e-9)

            outside_score = patch_2_region_avg * (unique_low_res_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((unique_low_res_masks == 0).sum(dim=0) + 1e-9)

            # Filter masks based on the difference score
            difference_score = (belong_score_avg - outside_score_avg).cpu().float().numpy()
            unique_low_res_masks[:, difference_score < self.global_patch_threshold] = 0

            if args.visualize_global_patch:
                plt.figure(figsize=(6, 6))
                plt.imshow(difference_score.reshape(self.points_per_h, self.points_per_w), cmap='coolwarm')
                plt.colorbar(label='Similarity')
                plt.title('Differences Within and Across Regions')
                plt.axis('off')
                plt.show()


        keep_masks = torch.sum(unique_low_res_masks, dim=1) > 0
        unique_low_res_masks = unique_low_res_masks[keep_masks]

        attn_weights = unique_low_res_masks.unsqueeze(0).repeat(num_heads, 1, 1)
        attn_weights = attn_weights.to(dtype=v_multi_head.dtype)

        attn_output = torch.bmm(attn_weights, v_multi_head)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)
        region_features = attn_output.permute(1, 0, 2)  # LND -> NLD

        region_features = self.clip.visual.ln_post(region_features) @ self.clip.visual.proj
        region_features /= region_features.norm(dim=-1, keepdim=True)
        return region_features, keep_masks


    def predict(self, args):
        img_arr = Image.open(args.image_dir).convert("RGB")
        img_arr = np.array(img_arr)

        if self.resize_method == 'multi_resolution':
            img_arr = imrescale(img_arr, (2016, 672), return_scale=False, interpolation='bilinear')
        else:
            raise NotImplementedError

        img_tensor = torch.from_numpy(img_arr).to(device="cuda", dtype=torch.float32)
        image_channel_first_float = img_tensor.permute(2, 0, 1) / 255.0

        ori_shape = img_arr.shape[:2]
        h, w = ori_shape[0], ori_shape[1]
        image_tensor_for_sam2 = torch.stack([img_tensor])
        image_tensor_for_sam2 = self.sam_transform(image_tensor_for_sam2)

        if self.resize_method == 'multi_resolution':
            clip_inputs = []
            if self.encode_global_image:
                clip_inputs.append(self.clip_preprocess(image_channel_first_float))

            self.crop_num_h, self.crop_num_w = max(h // self.crop_size, 1), max(w // self.crop_size, 1)
            self.points_per_w = (self.crop_size // self.patch_size) * self.crop_num_w
            self.points_per_h = (self.crop_size // self.patch_size) * self.crop_num_h
            crop_size_h, crop_size_w = int(np.ceil(h / self.crop_num_h)), int(np.ceil(w / self.crop_num_w))
            assert self.crop_num_h * crop_size_h >= h and self.crop_num_w * crop_size_w >= w

            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * crop_size_h
                    x1 = w_idx * crop_size_w
                    y2 = min(y1 + crop_size_h, h)
                    x2 = min(x1 + crop_size_w, w)
                    y1 = max(y2 - crop_size_h, 0)
                    x1 = max(x2 - crop_size_w, 0)
                    crop_img = image_channel_first_float[:, y1:y2, x1:x2]
                    clip_inputs.append(self.clip_preprocess(crop_img))


            clip_inputs = torch.stack(clip_inputs).to(self.device)

        else:
            raise NotImplementedError

        clip_inputs = clip_inputs.to(self.device, dtype=self.dtype)
        clip_last_blk_value, clip_last_blk = self.clip.encode_image(clip_inputs, return_value=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            sam2_masks = self.sam2_generator.generate_for_batch(image_tensor_for_sam2, [ori_shape], None)
        unique_masks = torch.stack([mask['segmentations'] for mask in sam2_masks[0]])

        unique_masks = unique_masks.to(self.device, dtype=self.dtype)
        unique_low_res_masks = F.interpolate(unique_masks.unsqueeze(0), [self.points_per_h, self.points_per_w], mode="bilinear")
        unique_low_res_masks = unique_low_res_masks.reshape(-1, self.points_per_h * self.points_per_w)
        unique_low_res_masks = torch.clamp(unique_low_res_masks, min=0, max=1)

        keep_masks = torch.sum(unique_low_res_masks, dim=1) > 0
        unique_low_res_masks = unique_low_res_masks[keep_masks]
        unique_masks = unique_masks[keep_masks]
        region_features, keep_masks = self.clip_value_with_sam2_attn(args, unique_low_res_masks, clip_last_blk_value, clip_last_blk)

        unique_masks = unique_masks[keep_masks]
        region_logits = region_features[0] @ self.query_features.T
        region_logits = region_logits.to(dtype=self.dtype)
        seg_logits, seg_preds = self.postprocess_result(region_logits, unique_masks, ori_shape)
        return seg_logits, seg_preds, img_arr, unique_masks


    def postprocess_result(self, region_logits, unique_masks, ori_shape):
        unique_masks = torch.clamp(unique_masks, min=0, max=1)
        seg_logits = region_logits.unsqueeze(-1).unsqueeze(-1) * unique_masks.unsqueeze(1)
        seg_logits = seg_logits.sum(0, keepdim=True)

        fill_logits = torch.zeros_like(seg_logits, device=self.device, dtype=seg_logits.dtype)
        fill_logits[:, 0, :, :] = 1.0  # background logits
        null_pixel = torch.nonzero(unique_masks.sum(0) == 0, as_tuple=True)
        seg_logits[:, :, null_pixel[0], null_pixel[1]] = fill_logits[:, :, null_pixel[0], null_pixel[1]]

        seg_logits = F.interpolate(seg_logits, size=ori_shape, mode='bilinear')
        seg_logits = torch.softmax(seg_logits * self.region_logit_scale, dim=1)

        seg_preds = seg_logits.argmax(1)
        seg_logits = seg_logits.max(1)[0]
        return seg_logits, seg_preds



def validate(clip_sam2_segmenter, segment_args, val_loader, args):

    split = args.val_dataset.split("|")[-1]
    if args.eval_query_type is None:
        eval_query_type_list = ["short", "long", "overall"]
    else:
        eval_query_type_list = [args.eval_query_type]

    eval_results = {}


    for eval_query_type in eval_query_type_list:
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

        for input_dict in tqdm(val_loader):
            torch.cuda.empty_cache()

            img_dir = input_dict["image_paths"][0]
            args.image_dir = segment_args.image_dir = img_dir

            img_name = os.path.basename(img_dir)
            json_name = img_name.replace(".jpg", ".json")
            json_path = os.path.join(args.interpreted_query_dir, json_name)

            try:
                with open(json_path, "r") as r:
                    anno = json.loads(r.read())
            except:
                with open(json_path, "r", encoding="cp1252") as r:
                    anno = json.loads(r.read())

            is_sentence = anno["is_sentence"]
            if split == "test" and eval_query_type == "short":
                if is_sentence:
                    continue
            elif split == "test" and eval_query_type == "long":
                if (not is_sentence):
                    continue

            input_dict = dict_to_cuda(input_dict)
            resize_list = input_dict["resize_list"][0]
            masks_list = input_dict["masks_list"][0]
            assert len(masks_list) == 1


            intersection, union, acc_iou = 0.0, 0.0, 0.0

            for i, mask_i in enumerate(masks_list):

                mask_i = mask_i.unsqueeze(0)
                if args.use_interpreted_query:
                    target = anno["answers"][i].lower()
                    if not is_sentence:
                        target = f'{target}, {anno["text"][i]}'.lower()
                else:
                    target = anno["text"][i].lower()


                if args.use_interpreted_query:
                    query_words = [f'{args.placeholder} {target}', target]
                else:
                    query_words = ['background, any other thing', target]
                _ = clip_sam2_segmenter.encode_query_words(query_words)

                seg_probs, seg_preds, img_ori, image = clip_sam2_segmenter.predict(segment_args)

                seg_preds = seg_preds.to(seg_preds.device, dtype=args.precision)
                mask_i = mask_i.int()
                if seg_preds.shape != mask_i.shape:
                    seg_preds = F.interpolate(seg_preds.unsqueeze(0), size=resize_list, mode="nearest")[0]
                output_i = (seg_preds > 0).int()

                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection), union_meter.update(
                union
            ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])


        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        print(f"Query: {eval_query_type}" + ", gIoU: {:.4f}, cIoU: {:.4f}".format(giou, ciou))
        eval_results[eval_query_type] = {
            "giou": giou,
            "ciou": ciou,
        }


    if args.eval_query_type is None:
        os.makedirs(args.log_dir, exist_ok=True)
        chicagoTz = pytz.timezone("America/Chicago")
        timeInChicago = datetime.now(chicagoTz)
        record_time = timeInChicago.strftime("%m_%d_%Y_%H_%M")
        log_name = f"{split}_{record_time}.txt"
        if args.use_interpreted_query:
            log_name = f"interpreted_{log_name}"

        results_dir = os.path.join(args.log_dir, log_name)
        with open(results_dir, "w") as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n\n")
            for key, value in vars(segment_args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n\n")

            for eval_query_type in eval_query_type_list:
                f.write(f"Query: {eval_query_type}" + ", gIoU: {:.4f}, cIoU: {:.4f}\n".format(
                    eval_results[eval_query_type]['giou'], eval_results[eval_query_type]['ciou']))
                f.write("\n")
    return eval_results



def main(args, segment_args):

    clip_sam2_segmenter = TextRegionReasonSegmenter(segment_args)
    args.title_text = f'CLIP_{segment_args.clip_architecture.replace("/", "_")}_resize_{segment_args.crop_size}'

    val_dataset = ValDataset(
        args.dataset_dir,
        clip_sam2_segmenter.clip_preprocess,
        args.val_dataset,
    )
    print(
        f"Validating with {len(val_dataset)} examples."
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        sampler=val_sampler,
        collate_fn=partial(
            collate_fn,
        ),
    )

    eval_results = validate(clip_sam2_segmenter, segment_args, val_loader, args)



def parse_args(cl_args: List[str] = None):
    parser = argparse.ArgumentParser(description="Evaluate ReasonSeg")

    parser.add_argument("--dataset_dir", default="/shared/nas2/yaox11/data/datasets", type=str)
    parser.add_argument("--interpreted_query_dir", type=str,
                        default="/shared/nas2/yaox11/data/datasets/reason_seg/interpreted_llava_v15_7b/test")

    parser.add_argument("--eval_query_type", choices=['short', 'long', 'overall', None], default=None)
    parser.add_argument("--use_interpreted_query", type=eval, choices=[True, False], default=True)
    parser.add_argument("--val_dataset", default="ReasonSeg|test", type=str)

    parser.add_argument("--log_dir", type=str, default="./eval_results/reason_seg")
    parser.add_argument("--placeholder", type=str, default="background, anything but")

    parser.add_argument("--clip_pretrained", type=str, default='openai')
    parser.add_argument("--clip_architecture", type=str, default='ViT-L/14@336px')
    return parser.parse_args(cl_args)



if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    cl_args = []
    for k, v in args.__dict__.items():
        arg_name = f'--{k}'
        if arg_name not in cl_args:
            cl_args.extend([arg_name, str(v)])


    segment_args = add_segment_config(cl_args)
    args.precision = segment_args.dtype

    with torch.inference_mode(), torch.autocast("cuda", dtype=args.precision):
        main(args, segment_args)
