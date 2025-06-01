###############################################################
## TextRegion                                                ##
## Copyright (c) 2025                                        ##
## Yao (Ava) Xiao                                            ##
## Licensed under the MIT License                            ##
## See project root for license details.                     ##
###############################################################


from sam2.build_sam import build_sam2
from sam2.custom_automatic_mask_generator import CustomAutomaticMaskGenerator
import custom_clip
from custom_open_clip import create_model, tokenizer
from custom_clip.clip import tokenize
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from utils.imagenet_template import openai_imagenet_template
from torchvision.transforms import v2
from utils.visualize_segmentation import *

import socket
import argparse
from typing import List
from mmcv.image import imrescale
from torchvision.transforms import InterpolationMode

from tqdm import tqdm
import yaml
import numpy as np
import torch.nn.functional as F
import torch



def parse_segment_args(cl_args: List[str] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--clip_download_root", type=str, help="Path to the CLIP download root.", default=None)
    parser.add_argument("--sam2_checkpoint", type=str, help="Path to the SAM2 checkpoint.", default=None)
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")

    parser.add_argument("--image_query_cfg", type=str, default="./utils/image_query_label.yaml")
    parser.add_argument("--image_list", nargs='+',
                        default=['./assets/dino_two_dogs.jpg', './assets/sam2_truck.jpg', './assets/boats_ambulance.jpg'])

    parser.add_argument("--clip_pretrained", type=str, choices=['openai', 'meta', 'siglip2'], default='openai')
    parser.add_argument("--clip_architecture", type=str, default='ViT-L/14@336px',
                        choices=['ViT-L/14@336px', 'PE-Core-L14-336', 'ViT-L-16-SigLIP2-256'])

    parser.add_argument("--remove_global_patch", type=eval, default=True)
    parser.add_argument("--visualize_global_patch", type=eval, default=False)
    parser.add_argument("--global_patch_threshold", type=float, default=None)

    parser.add_argument("--resize_method", choices=['multi_resolution', 'resize'], default='multi_resolution')
    parser.add_argument("--scale", nargs='+', default=[2016, 672])
    parser.add_argument("--crop_size", type=int, default=336)

    parser.add_argument("--points_per_side", type=int, default=16)
    parser.add_argument("--dtype", type=str, default='fp32', choices=['bf16', 'fp32'])

    args, unknown = parser.parse_known_args(cl_args)
    return args


def add_segment_config(cl_args=[]):

    args = parse_segment_args(cl_args)
    if args.dtype == "fp32":
        args.dtype = torch.float32
    elif args.dtype == "bf16":
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        args.dtype = torch.bfloat16 if use_bf16 else torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    print(f"Using dtype: {args.dtype}")
    return args


def load_sam2(sam2_cfg, model_cfg, sam2_checkpoint):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2_generator = CustomAutomaticMaskGenerator(
        prompt_method="grid",
        model=sam2_model,
        point_grids=None,
        min_mask_region_area=0,
        points_per_side=sam2_cfg.points_per_side,
        points_per_batch=2048,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.6,
        box_nms_thresh=0.9,
        multimask_output=True,
        fuse_mask=True,
        fuse_mask_threshold=0.8,
    )

    print(f'Finish loading sam2.1, and start predicting images\n')
    return sam2_generator




class CLIPWithSAM2ForSegmentation():
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
                # v2.ToImage(),
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

        elif self.clip_pretrained == 'meta':
            self.clip = pe.CLIP.from_config(args.clip_architecture, pretrained=True)
            self.clip.eval().to(device)
            self.patch_size = self.clip.visual.patch_size
            self.tokenizer = transforms.get_text_tokenizer(self.clip.context_length)
            self.crop_size = self.clip.visual.image_size

            clip_preprocess = v2.Compose(
                [
                    v2.Resize((self.crop_size, self.crop_size), interpolation=InterpolationMode.BILINEAR),
                    v2.ToDtype(self.dtype, scale=True),
                    v2.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                    ),
                ]
            )

        elif self.clip_pretrained == 'siglip2':
            import open_clip
            model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_architecture, pretrained='webli')
            model = model.to(device=device, dtype=self.dtype)
            self.tokenizer = open_clip.get_tokenizer(args.clip_architecture)
            self.text_model = model.text
            self.patch_size = model.visual.trunk.patch_embed.patch_size[0]
            self.clip = model.eval()
            self.crop_size = model.visual.image_size[0]

            clip_preprocess = v2.Compose(
                [
                    v2.Resize((self.crop_size, self.crop_size), interpolation=InterpolationMode.BICUBIC),
                    v2.ToDtype(self.dtype, scale=True),
                    v2.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                    ),
                ]
            )

        else:
            self.clip = create_model(args.clip_architecture, pretrained=args.clip_pretrained, precision=args.dtype, cache_dir=args.clip_download_root)
            self.clip.eval().to(device)
            self.patch_size = self.clip.visual.patch_size[0]

        self.clip_preprocess = clip_preprocess

        self.points_per_side = args.points_per_side
        self.sam2_generator = load_sam2(args, args.model_cfg, args.sam2_checkpoint)
        self.sam_transform = self.sam2_generator.predictor._transforms


    def encode_query_words(self, query_words):
        if type(query_words) == str:
            query_words = [query_words]

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                if self.clip_pretrained in ['meta', 'siglip2']:
                    qw_list = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                else:
                    qw_list = tokenize([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                feature = self.clip.encode_text(qw_list)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        query_features = torch.cat(query_features, dim=0)
        self.query_features = query_features.to(self.device, dtype=self.dtype)
        return self.query_features


    def get_colormap(self, class_names):
        colormap = []
        hues = np.linspace(0, 180, len(class_names), endpoint=False)

        for hue, class_name in zip(hues, class_names):
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0].tolist()
            colormap.append(color)
        return colormap


    def siglip_value_with_sam2_attn(self, args, low_res_mask_with_pad, last_blk_value, attn_blk):
        bsz, _, embed_dim = last_blk_value.shape
        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            x_ori = last_blk_value.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)

            crop_id = 1
            x_multi_reso = F.interpolate(x_ori[:1], [self.points_per_h, self.points_per_w], mode="bilinear")
            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num

                    x_multi_reso[:, :, y1:y2, x1:x2] = 0.5 * x_multi_reso[:, :, y1:y2, x1:x2] + x_ori[crop_id]
                    crop_id += 1

            x_input = x_multi_reso.contiguous().view(1, embed_dim, self.crop_num_h * self.crop_num_w * patch_num ** 2).permute(0, 2, 1)

        else:
            x_input = last_blk_value

        if args.remove_global_patch:

            keep_masks = torch.sum(low_res_mask_with_pad, dim=1) > 0
            low_res_mask = low_res_mask_with_pad[keep_masks]

            patch_norm = x_input.norm(dim=-1, keepdim=True)
            patch_features = (x_input / patch_norm)[0]
            patch_similarity = (patch_features @ patch_features.T).float()

            patch_2_region = patch_similarity @ (low_res_mask > 0).float().T
            patch_2_region_avg = patch_2_region / (low_res_mask > 0).sum(dim=-1)

            blong_score = patch_2_region_avg * (low_res_mask > 0).float().T
            blong_score_avg = blong_score.sum(dim=-1) / ((low_res_mask > 0).sum(dim=0) + 1e-9)

            outside_score = patch_2_region_avg * (low_res_mask == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((low_res_mask == 0).sum(dim=0) + 1e-9)

            difference_score = (blong_score_avg - outside_score_avg).cpu().float().numpy()

            if args.visualize_global_patch:
                plt.figure(figsize=(6, 6))
                plt.imshow(difference_score.reshape(self.points_per_h, self.points_per_w), cmap='coolwarm')
                plt.colorbar(label='Similarity')
                plt.title('Differences Within and Across Regions')
                plt.axis('off')
                plt.show()

            # Set the threshold for the difference score
            low_res_mask_with_pad[:, difference_score < self.global_patch_threshold] = 0

        keep_masks = torch.sum(low_res_mask_with_pad, dim=1) > 0
        low_res_mask_with_pad = low_res_mask_with_pad[keep_masks]
        low_res_mask_with_pad = torch.clamp(low_res_mask_with_pad, min=0, max=1)

        assert x_input.shape[0] == 1
        region_num = low_res_mask_with_pad.shape[0]

        # Modify by @Ava: based on timm/layers/attention_pool.py
        _, N, C = x_input.shape
        q_latent = attn_blk.latent.expand(region_num, -1, -1)
        q = attn_blk.q(q_latent).reshape(region_num, attn_blk.latent_len, attn_blk.num_heads, attn_blk.head_dim).transpose(1, 2)

        x = x_input.expand(region_num, -1, -1)
        kv = attn_blk.kv(x).reshape(region_num, N, 2, attn_blk.num_heads, attn_blk.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = attn_blk.q_norm(q), attn_blk.k_norm(k)

        attn_mask = low_res_mask_with_pad.unsqueeze(1).unsqueeze(1).repeat(1, attn_blk.num_heads, 1, 1)

        k = attn_blk.k_norm(k.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True))
        k = k.repeat(1, 1, v.shape[-2], v.shape[-1])
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask > 0)

        x = x.transpose(1, 2).reshape(region_num, attn_blk.latent_len, C)
        x = attn_blk.proj(x)
        x = attn_blk.proj_drop(x)

        x = self.clip.visual.trunk.fc_norm(x)
        x = self.clip.visual.trunk.head_drop(x)

        region_features = x.permute(1, 0, 2)
        region_features /= region_features.norm(dim=-1, keepdim=True)
        return region_features, keep_masks


    def pe_value_with_sam2_attn(self, args, unique_low_res_masks, last_blk_value, blk):

        if self.clip.visual.use_cls_token:
            last_blk_value = last_blk_value[:, 1:]

        bsz, _, embed_dim = last_blk_value.shape
        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            x_ori = last_blk_value.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)

            crop_id = 1
            x_multi_reso = F.interpolate(x_ori[:1], [self.points_per_h, self.points_per_w], mode="bilinear")
            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num

                    x_multi_reso[:, :, y1:y2, x1:x2] = 0.5 * x_multi_reso[:, :, y1:y2, x1:x2] + x_ori[crop_id]
                    crop_id += 1

            x_input = x_multi_reso.contiguous().view(1, embed_dim, self.points_per_h * self.points_per_w).permute(0, 2, 1)
        else:
            x_input = last_blk_value

        if args.remove_global_patch:
            patch_norm = x_input.norm(dim=-1, keepdim=True)
            patch_features = (x_input / patch_norm)[0]
            patch_similarity = patch_features @ patch_features.T

            patch_2_region = patch_similarity @ (unique_low_res_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (unique_low_res_masks > 0).sum(dim=-1)

            belong_score = patch_2_region_avg * (unique_low_res_masks > 0).float().T
            belong_score_avg = belong_score.sum(dim=-1) / ((unique_low_res_masks > 0).sum(dim=0) + 1e-9)

            outside_score = patch_2_region_avg * (unique_low_res_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((unique_low_res_masks == 0).sum(dim=0) + 1e-9)

            difference_score = (belong_score_avg - outside_score_avg).cpu().float().numpy()

            if args.visualize_global_patch:
                plt.figure(figsize=(6, 6))
                plt.imshow(difference_score.reshape(self.points_per_h, self.points_per_w), cmap='coolwarm')
                plt.colorbar(label='Similarity')
                plt.title(f'In-and-Out Region Similarity')
                plt.axis('off')
                plt.show()

            # Set the threshold for the difference score
            unique_low_res_masks[:, difference_score < self.global_patch_threshold] = 0

        keep_masks = torch.sum(unique_low_res_masks, dim=1) > 0
        unique_low_res_masks = unique_low_res_masks[keep_masks]
        batch = unique_low_res_masks.shape[0]


        assert x_input.shape[0] == 1 or x_input.shape[0] == unique_low_res_masks.shape[0]
        if x_input.shape[0] == 1:
            x = x_input.repeat(batch, 1, 1)
        else:
            x = x_input

        q = blk.probe.repeat((batch, 1, 1)).to(x.dtype)
        k = blk.layernorm(x.mean(dim=-2, keepdim=True))
        k = k.repeat(1, x.shape[-2], 1).to(x.dtype)
        x = blk.attn(q, k, x, need_weights=False, key_padding_mask=unique_low_res_masks<=0)[0]

        with torch.no_grad():
            region_features =  x @ self.clip.visual.proj
        region_features = F.normalize(region_features, dim=-1)
        return region_features, keep_masks


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
            v_single_head = v_ori[1:].permute(1, 2, 0)
            v_multi_head = v_ori[1:].contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

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
        attn_output += blk.mlp(blk.ln_2(attn_output))
        region_features = attn_output.permute(1, 0, 2)  # LND -> NLD

        region_features = self.clip.visual.ln_post(region_features) @ self.clip.visual.proj
        region_features /= region_features.norm(dim=-1, keepdim=True)
        return region_features, keep_masks


    def predict(self, args, compute_logits=True, return_region_logits=False):
        if return_region_logits:
            assert compute_logits

        img_arr = Image.open(args.image_dir).convert("RGB")
        img_arr = np.array(img_arr)

        if self.resize_method == 'multi_resolution':
            img_arr = imrescale(img_arr, (args.scale[0], args.scale[1]), return_scale=False, interpolation='bilinear')
        else:
            img_arr = cv2.resize(img_arr, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

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
            self.points_per_w = w // self.patch_size
            self.points_per_h = h // self.patch_size
            clip_inputs = self.clip_preprocess(image_channel_first_float).unsqueeze(0)


        if self.clip_pretrained == 'meta':
            clip_inputs = clip_inputs.to(self.device, dtype=self.clip.visual.proj.dtype)
            pe_last_blk_value, pe_last_blk = self.clip.encode_image(clip_inputs, return_value=True, region_attn_mask=None)
        elif self.clip_pretrained == 'siglip2':
            siglip_last_blk_value, intermediates = self.clip.visual.trunk.forward_intermediates(clip_inputs)
            siglip_last_blk = self.clip.visual.trunk.attn_pool
        else:
            clip_inputs = clip_inputs.to(self.device, dtype=self.clip.visual.proj.dtype)
            clip_last_blk_value, clip_last_blk = self.clip.encode_image(clip_inputs, return_value=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
            sam2_masks = self.sam2_generator.generate_for_batch(image_tensor_for_sam2, [ori_shape], None)
        unique_masks = torch.stack([mask['segmentations'] for mask in sam2_masks[0]])


        unique_masks = unique_masks.to(self.device, dtype=self.dtype)
        unique_low_res_masks = F.interpolate(unique_masks.unsqueeze(0), [self.points_per_h, self.points_per_w], mode="bilinear")
        unique_low_res_masks = unique_low_res_masks.reshape(-1, self.points_per_h * self.points_per_w)
        unique_low_res_masks = torch.clamp(unique_low_res_masks, min=0, max=1)

        keep_masks = torch.sum(unique_low_res_masks, dim=1) > 0
        unique_low_res_masks = unique_low_res_masks[keep_masks]
        unique_masks = unique_masks[keep_masks]


        if self.clip_pretrained == 'meta':
            region_features, keep_masks = self.pe_value_with_sam2_attn(args, unique_low_res_masks, pe_last_blk_value, pe_last_blk)
            region_features = region_features.permute(1, 0, 2)
        elif self.clip_pretrained == 'siglip2':
            region_features, keep_masks = self.siglip_value_with_sam2_attn(args, unique_low_res_masks, siglip_last_blk_value, siglip_last_blk)
        else:
            region_features, keep_masks = self.clip_value_with_sam2_attn(args, unique_low_res_masks, clip_last_blk_value, clip_last_blk)


        unique_masks = unique_masks[keep_masks]
        if compute_logits:
            if self.clip_pretrained == 'siglip2':
                logits_per_text = (
                        torch.matmul(self.query_features, region_features[0].t()) * self.clip.logit_scale.exp()
                        + self.clip.logit_bias
                )
                region_logits = logits_per_text.t()
            else:
                region_logits = region_features[0] @ self.query_features.T

            region_logits = region_logits.to(dtype=self.dtype)
            seg_logits, seg_preds = self.postprocess_result(region_logits, unique_masks, ori_shape)

            if return_region_logits:
                return seg_logits, seg_preds, region_logits, img_arr, unique_masks
            else:
                return seg_logits, seg_preds, img_arr, unique_masks

        else:
            return region_features, unique_masks


    def postprocess_result(self, region_logits, unique_masks, ori_shape):
        unique_masks = torch.clamp(unique_masks, min=0, max=1)
        seg_logits = region_logits.unsqueeze(-1).unsqueeze(-1) * unique_masks.unsqueeze(1)
        seg_logits = seg_logits.sum(0, keepdim=True)

        seg_logits = F.interpolate(seg_logits, size=ori_shape, mode='bilinear')
        seg_logits = torch.softmax(seg_logits * self.region_logit_scale, dim=1)

        seg_preds = seg_logits.argmax(1)
        seg_logits = seg_logits.max(1)[0]
        return seg_logits, seg_preds



def main(args):
    clip_sam2_segmenter = CLIPWithSAM2ForSegmentation(args)

    with open(args.image_query_cfg, "r") as f:
        image_query_dict = yaml.safe_load(f)

    for img_dir in tqdm(args.image_list):
        args.image_dir = img_dir
        query_words = image_query_dict[img_dir]['label']
        clip_sam2_segmenter.encode_query_words(query_words)
        colormap = clip_sam2_segmenter.get_colormap(query_words)

        seg_logits, seg_preds, region_logits, image, unique_masks = clip_sam2_segmenter.predict(args, return_region_logits=True)

        for seg_pred, seg_logit, unique_mask in zip(seg_preds, seg_logits, unique_masks):
            seg_pred = seg_pred.cpu().numpy()
            render_segmentation(query_words, colormap, seg_pred, image, show_img=True)

    return



if __name__ == '__main__':

    args = add_segment_config(sys.argv[1:])
    with torch.inference_mode(), torch.autocast("cuda", dtype=args.dtype):
        main(args)