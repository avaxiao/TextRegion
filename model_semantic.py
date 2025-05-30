###############################################################
## TextRegion                                                ##
## Copyright (c) 2025                                        ##
## Yao (Ava) Xiao                                            ##
## Licensed under the Apache License, Version 2.0            ##
## See project root for license details.                     ##
###############################################################


import sys
sys.path.append("..")
sys.path.append(".")

from sam2.build_sam import build_sam2
from sam2.custom_automatic_mask_generator import CustomAutomaticMaskGenerator
import custom_clip
import custom_open_clip
import open_clip
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from utils.imagenet_template import openai_imagenet_template
from torchvision.transforms import v2
from custom_clip.clip import tokenize
from utils.visualize_segmentation import *
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
import warnings

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode



def load_sam2(sam2_cfg, sam2_architecture, model_cfg, sam2_checkpoint):
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
        model=sam2_model,
        point_grids=None,
        points_per_side=sam2_cfg.points_per_side,
        points_per_batch=2048,
        pred_iou_thresh=sam2_cfg.pred_iou_thresh,
        stability_score_thresh=sam2_cfg.stability_score_thresh,
        box_nms_thresh=sam2_cfg.box_nms_thresh,
        multimask_output=True,
        fuse_mask=sam2_cfg.fuse_mask,
        fuse_mask_threshold=sam2_cfg.fuse_mask_threshold,
    )
    print(f'Finish loading sam2.1 {sam2_architecture}, and start predicting images\n')
    return sam2_generator



@MODELS.register_module()
class CLIPWithSAM2ForSegmentation(BaseSegmentor):
    def __init__(self,
                 name_path,
                 clip_pretrained,
                 clip_architecture,
                 clip_download_root,
                 sam2_architecture,
                 sam2_checkpoint,
                 model_cfg,
                 config=None,
                 resize_method='multi_resolution',
                 remove_global_patch=True,
                 global_patch_threshold=0.07,
                 sam2_cfg=None,
                 crop_size=336,
                 region_logit_scale=50,
                 prob_thd=0.0,
                 upsample_times=1,
                 mask_type='soft',
                 pool_feature='value',
                 dtype='fp32',
                 device = torch.device("cuda") if torch.cuda.is_available() else "cpu",
    ):

        data_preprocessor = SegDataPreProcessor(
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        if dtype == "fp32":
            dtype = torch.float32
        elif dtype == "bf16":
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.device = device
        self.dtype = dtype

        self.upsample_times = upsample_times
        self.mask_type = mask_type
        self.pool_feature = pool_feature

        self.resize_method = resize_method
        self.crop_size = crop_size
        self.region_logit_scale = region_logit_scale
        self.prob_thd = prob_thd
        self.config = config
        self.fuse_mask = sam2_cfg.fuse_mask
        self.fuse_mask_threshold = sam2_cfg.fuse_mask_threshold
        self.remove_global_patch = remove_global_patch
        self.global_patch_threshold = global_patch_threshold
        self.clip_pretrained = clip_pretrained


        clip_preprocess = v2.Compose(
            [
                v2.Resize((crop_size, crop_size), interpolation=Image.BICUBIC),
                v2.ToDtype(self.dtype, scale=True),
                v2.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.clip_pretrained = clip_pretrained
        if self.clip_pretrained in ['laion2b_s32b_b79k', 'siglip2', 'meta']:
            assert self.upsample_times == 1
            assert self.mask_type == 'soft'
            assert self.pool_feature == 'value'


        if clip_pretrained == 'openai':
            self.clip, _ = custom_clip.load(clip_architecture, device=device, jit=False, download_root=clip_download_root)
            self.patch_size = self.clip.visual.patch_size

        elif clip_pretrained == 'laion2b_s32b_b79k':
            self.clip = custom_open_clip.create_model(clip_architecture, pretrained=clip_pretrained, precision=dtype, cache_dir=clip_download_root)
            self.clip.eval().to(device)
            self.patch_size = self.clip.visual.patch_size[0]
            self.clip_text_tokenizer = custom_open_clip.tokenizer.tokenize

        elif clip_pretrained == 'siglip2':
            model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_architecture, pretrained='webli')
            model = model.to(device=device, dtype=self.dtype)
            self.tokenizer = open_clip.get_tokenizer(clip_architecture)
            self.text_model = model.text
            self.logit_scale = model.logit_scale
            self.logit_bias = model.logit_bias
            self.patch_size = model.visual.trunk.patch_embed.patch_size[0]
            self.clip = model.eval()
            self.crop_size = model.visual.image_size[0]
            self.clip = self.clip.to(dtype=self.dtype)

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

        elif clip_pretrained == 'meta':
            self.clip = pe.CLIP.from_config(clip_architecture, pretrained=True)
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

        self.clip_preprocess = clip_preprocess
        self.encode_text(name_path)

        self.points_per_side = sam2_cfg.points_per_side
        self.sam2_generator = load_sam2(sam2_cfg, sam2_architecture, model_cfg, sam2_checkpoint)
        self.sam_transform = self.sam2_generator.predictor._transforms


    def encode_text(self, name_path):
        query_words, self.query_idx = get_cls_idx(name_path)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                if self.clip_pretrained in ['openai', 'laion2b_s32b_b79k']:
                    query = tokenize([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                    feature = self.clip.encode_text(query)
                else:
                    text_inputs = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                    feature = self.clip.encode_text(text_inputs)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.num_cls, self.num_queries = max(self.query_idx) + 1, len(self.query_idx)
        if self.num_cls == self.num_queries:
            self.class_names = query_words
        else:
            self.class_names = ['background'] + query_words[-self.num_cls + 1:]

        if self.prob_thd != 0.0 and self.num_cls == self.num_queries and self.class_names[0] != 'background':
            warnings.warn("prob_thd is non-zero, but the first label is not 'background' or similar. Set prob_thd to zero if no label is named 'background' or something similar.", UserWarning)


    def siglip_value_with_sam2_attn(self, feature_masks, input_feature, attn_blk, query_features, logit_scale, logit_bias):

        bsz, _, embed_dim = input_feature.shape
        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            x_ori = input_feature.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)

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
            x_input = input_feature

        if self.remove_global_patch:
            patch_norm = x_input.norm(dim=-1, keepdim=True)
            patch_features = (x_input / patch_norm)[0]
            patch_similarity = (patch_features @ patch_features.T).float()

            patch_2_region = patch_similarity @ (feature_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (feature_masks > 0).sum(dim=-1)

            blong_score = patch_2_region_avg * (feature_masks > 0).float().T
            blong_score_avg = blong_score.sum(dim=-1) / ((feature_masks > 0).sum(dim=0) + 1e-9)

            outside_score = patch_2_region_avg * (feature_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((feature_masks == 0).sum(dim=0) + 1e-9)

            difference_score = (blong_score_avg - outside_score_avg).cpu().float().numpy()

            # Set the threshold for the difference score
            # threshold = difference_score[difference_score > 0].mean()
            feature_masks[:, difference_score < self.global_patch_threshold] = 0

        keep_masks = torch.sum(feature_masks, dim=1) > 0
        feature_masks = feature_masks[keep_masks]

        assert x_input.shape[0] == 1
        region_num = feature_masks.shape[0]

        # Modify by @Ava: based on timm/layers/attention_pool.py
        _, N, C = x_input.shape
        q_latent = attn_blk.latent.expand(region_num, -1, -1)
        q = attn_blk.q(q_latent).reshape(region_num, attn_blk.latent_len, attn_blk.num_heads, attn_blk.head_dim).transpose(1, 2)

        x = x_input.expand(region_num, -1, -1)
        kv = attn_blk.kv(x).reshape(region_num, N, 2, attn_blk.num_heads, attn_blk.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = attn_blk.q_norm(q), attn_blk.k_norm(k)

        attn_mask = feature_masks.unsqueeze(1).unsqueeze(1).repeat(1, attn_blk.num_heads, 1, 1)
        k = attn_blk.k_norm(k.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True))
        k = k.repeat(1, 1, v.shape[-2], v.shape[-1])
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask > 0)

        x = x.transpose(1, 2).reshape(region_num, attn_blk.latent_len, C)
        x = attn_blk.proj(x)
        x = attn_blk.proj_drop(x)

        x = self.clip.visual.trunk.fc_norm(x)
        x = self.clip.visual.trunk.head_drop(x)

        image_features = x[:, 0]
        image_features /= image_features.norm(dim=-1, keepdim=True)
        query_features = query_features.to(dtype=image_features.dtype)
        logits_per_text = (
            torch.matmul(query_features, image_features.t()) * logit_scale.exp()
            + logit_bias
        )
        region_logits = logits_per_text.t()
        return region_logits, keep_masks


    def pe_value_with_sam2_attn(self, feature_masks, input_feature, blk, query_features):

        if self.clip.visual.use_cls_token:
            input_feature = input_feature[:, 1:]


        bsz, _, embed_dim = input_feature.shape
        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            x_ori = input_feature.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)

            crop_id = 1
            x_multi_reso = F.interpolate(x_ori[:1], [self.points_per_h, self.points_per_w], mode="bilinear")
            for h_idx in range(self.crop_num_h):
                for w_idx in range (self.crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num

                    x_multi_reso[:, :, y1:y2, x1:x2] = 0.5 * x_multi_reso[:, :, y1:y2, x1:x2] + x_ori[crop_id]
                    crop_id += 1

            x_input = x_multi_reso.contiguous().view(1, embed_dim, self.crop_num_h * self.crop_num_w * patch_num ** 2).permute(0, 2, 1)
        else:
            x_input = input_feature

        if self.remove_global_patch:
            feature_masks_ori = feature_masks.clone()
            patch_norm = x_input.norm(dim=-1, keepdim=True)
            patch_features = (x_input / patch_norm)[0]
            patch_similarity = patch_features @ patch_features.T

            patch_2_region = patch_similarity @ (feature_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (feature_masks > 0).sum(dim=-1)

            belong_score = patch_2_region_avg * (feature_masks > 0).float().T
            belong_score_avg = belong_score.sum(dim=-1) / ((feature_masks > 0).sum(dim=0) + 1e-9)

            outside_score = patch_2_region_avg * (feature_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((feature_masks == 0).sum(dim=0) + 1e-9)

            difference_score = (belong_score_avg - outside_score_avg).cpu().float().numpy()

            # Set the threshold for the difference score
            # threshold = difference_score[difference_score > 0].mean()
            feature_masks[:, difference_score < self.global_patch_threshold] = 0


        keep_masks = torch.sum(feature_masks, dim=1) > 0
        if keep_masks.sum() == 0:
            feature_masks = feature_masks_ori
            keep_masks = torch.sum(feature_masks, dim=1) == 0
        else:
            feature_masks = feature_masks[keep_masks]
        batch = feature_masks.shape[0]

        assert x_input.shape[0] == 1 or x_input.shape[0] == feature_masks.shape[0]
        if x_input.shape[0] == 1:
            x = x_input.repeat(batch, 1, 1)
        else:
            x = x_input

        q = blk.probe.repeat((batch, 1, 1)).to(x.dtype)
        k = blk.layernorm(x.mean(dim=-2, keepdim=True))
        k = k.repeat(1, x.shape[-2], 1).to(x.dtype)
        x = blk.attn(q, k, x, need_weights=False, key_padding_mask=feature_masks<=0)[0]

        with torch.no_grad():
            region_features =  x @ self.clip.visual.proj
        region_features = F.normalize(region_features, dim=-1)

        query_features = query_features.to(dtype=region_features.dtype)
        region_logits = region_features.squeeze(1) @ query_features.T

        return region_logits, keep_masks


    def clip_value_with_sam2_attn(self, feature_masks, input_feature, blk, query_features):
        attn_layer = blk.attn
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = input_feature.size()
        head_dim = embed_dim // num_heads

        x = blk.ln_1(input_feature)
        q, k, v_ori = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)


        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size

            crop_id = 1
            v = v_ori[1:, :, :].permute(1, 2, 0).contiguous().view(bsz, embed_dim, patch_num, patch_num)

            if self.upsample_times > 1:
                patch_num *= self.upsample_times
                v = F.interpolate(v, scale_factor=self.upsample_times, mode="bilinear", align_corners=False)


            if "city_scape" in self.config:
                v_multi_reso_left = F.interpolate(v[0].unsqueeze(0), [self.points_per_h, self.points_per_w//2], mode="bilinear",
                                         align_corners=False)
                v_multi_reso_right = F.interpolate(v[1].unsqueeze(0), [self.points_per_h, self.points_per_w//2], mode="bilinear",
                                     align_corners=False)
                v_multi_reso = torch.cat((v_multi_reso_left,v_multi_reso_right),dim=-1)
                crop_id = 2
            else:
                v_multi_reso = F.interpolate(v[:1], [self.points_per_h, self.points_per_w], mode="bilinear")

            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num

                    v_multi_reso[:, :, y1:y2, x1:x2] = v_multi_reso[:, :, y1:y2, x1:x2] + v[crop_id]
                    crop_id += 1

            bsz = 1
            v_single_head = v_multi_reso.contiguous().view(1, embed_dim, self.points_per_h * self.points_per_w)
            v_multi_head = v_single_head.contiguous().view(bsz * num_heads, head_dim, self.points_per_h * self.points_per_w).permute(0, 2, 1)

        else:
            v_single_head = v_ori[1:, :, :].permute(1, 2, 0)
            v_multi_head = v_ori.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)[:, 1:]

        if self.remove_global_patch:
            feature_masks = feature_masks.float()
            patch_features = v_single_head.permute(0, 2, 1)[0].to(feature_masks.dtype)
            patch_features /= patch_features.norm(dim=-1, keepdim=True)
            patch_similarity = patch_features @ patch_features.T

            patch_2_region = patch_similarity @ (feature_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (feature_masks > 0).sum(dim=-1)

            belong_score = patch_2_region_avg * (feature_masks > 0).float().T
            belong_score_avg = belong_score.sum(dim=-1) / ((feature_masks > 0).sum(dim=0) + 1e-9)

            outside_score = patch_2_region_avg * (feature_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((feature_masks == 0).sum(dim=0) + 1e-9)

            # Filter masks based on the difference score
            difference_score = (belong_score_avg - outside_score_avg).cpu().float().numpy()
            feature_masks[:, difference_score < self.global_patch_threshold] = 0

        keep_masks = torch.sum(feature_masks, dim=1) > 0
        feature_masks = feature_masks[keep_masks]

        attn_weights = feature_masks.unsqueeze(0).repeat(num_heads, 1, 1)
        attn_weights = attn_weights.to(dtype=v_multi_head.dtype)

        attn_output = torch.bmm(attn_weights, v_multi_head)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)
        image_features = attn_output.permute(1, 0, 2)  # LND -> NLD

        image_features = self.clip.visual.ln_post(image_features) @ self.clip.visual.proj
        image_features /= image_features.norm(dim=-1, keepdim=True)

        query_features = query_features.to(image_features.dtype)
        region_logits = image_features[0] @ query_features.T

        return region_logits, keep_masks


    def clip_feature_with_sam2(self, feature_masks, input_feature, query_features):

        _, bsz, embed_dim = input_feature.shape

        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size

            crop_id = 1
            v = input_feature[1:, :, :].permute(1, 2, 0).contiguous().view(bsz, embed_dim, patch_num, patch_num)

            if self.upsample_times > 1:
                patch_num *= self.upsample_times

            if "city_scape" in self.config:
                v_multi_reso_left = F.interpolate(v[0].unsqueeze(0), [self.points_per_h, self.points_per_w//2], mode="bilinear",
                                         align_corners=False)
                v_multi_reso_right = F.interpolate(v[1].unsqueeze(0), [self.points_per_h, self.points_per_w//2], mode="bilinear",
                                     align_corners=False)
                v_multi_reso = torch.cat((v_multi_reso_left,v_multi_reso_right),dim=-1)
                crop_id = 2
            else:
                v_multi_reso = F.interpolate(v[:1], [self.points_per_h, self.points_per_w], mode="bilinear")

            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num

                    v_multi_reso[:, :, y1:y2, x1:x2] = v_multi_reso[:, :, y1:y2, x1:x2] + v[crop_id]
                    crop_id += 1

            v_single_head = v_multi_reso.contiguous().view(1, embed_dim, self.points_per_h * self.points_per_w)
            v_multi_head = v_multi_reso.contiguous().view(1, embed_dim, self.points_per_h * self.points_per_w).permute(0, 2, 1)

        else:
            raise ValueError("Only support multi-resolution")

        if self.remove_global_patch:
            feature_masks = feature_masks.float()
            patch_features = v_single_head.permute(0, 2, 1)[0].to(feature_masks.dtype)
            patch_features /= patch_features.norm(dim=-1, keepdim=True)
            patch_similarity = patch_features @ patch_features.T

            patch_2_region = patch_similarity @ (feature_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (feature_masks > 0).sum(dim=-1)

            belong_score = patch_2_region_avg * (feature_masks > 0).float().T
            belong_score_avg = belong_score.sum(dim=-1) / ((feature_masks > 0).sum(dim=0) + 1e-9)

            outside_score = patch_2_region_avg * (feature_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((feature_masks == 0).sum(dim=0) + 1e-9)

            # Filter masks based on the difference score
            difference_score = (belong_score_avg - outside_score_avg).cpu().float().numpy()
            feature_masks[:, difference_score < self.global_patch_threshold] = 0

        keep_masks = torch.sum(feature_masks, dim=1) > 0
        feature_masks = feature_masks[keep_masks]

        region_feature = feature_masks.to(self.dtype) @ v_multi_head[0].to(self.dtype)
        region_feature = region_feature.to(self.clip.visual.proj.dtype)
        image_features = self.clip.visual.ln_post(region_feature) @ self.clip.visual.proj
        image_features /= image_features.norm(dim=-1, keepdim=True)

        query_features = query_features.to(image_features.dtype)
        region_logits = image_features @ query_features.T
        return region_logits, keep_masks


    def get_CLIP_logits(self, input_feature, blk, query_features):
        x = input_feature
        x = x + self.clip.visual.custom_attn(blk.attn, blk.ln_1(x), csa=True)
        x = x + blk.mlp(blk.ln_2(x))
        x = x.permute(1, 0, 2)  # LND -> NLD
        clip_outputs = self.clip.visual.ln_post(x) @ self.clip.visual.proj
        clip_outputs /= clip_outputs.norm(dim=-1, keepdim=True)
        query_features = query_features.to(clip_outputs.dtype)
        clip_logits = clip_outputs[:, 1:] @ query_features.T
        patch_num = self.crop_size // self.patch_size

        if self.resize_method == 'multi_resolution':
            bsz, _, class_num = clip_logits.size()
            clip_logits = clip_logits.permute(0, 2, 1).contiguous().view(bsz, -1, patch_num, patch_num)

            crop_id = 1

            if "city_scape" in self.config:
                logit_resize_left = F.interpolate(clip_logits[0].unsqueeze(0), [self.points_per_h, self.points_per_w//2], mode="bilinear",
                                         align_corners=False)
                logit_resize_right = F.interpolate(clip_logits[1].unsqueeze(0), [self.points_per_h, self.points_per_w//2], mode="bilinear",
                                     align_corners=False)
                logit_resize = torch.cat((logit_resize_left, logit_resize_right), dim=-1)
                crop_id = 2
            else:
                logit_resize = F.interpolate(clip_logits[:1], [self.points_per_h, self.points_per_w],
                                             mode="bilinear", align_corners=False)


            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num
                    logit_resize[:, :, y1:y2, x1:x2] = logit_resize[:, :, y1:y2, x1:x2] + clip_logits[crop_id]
                    crop_id += 1

        else:
            logit_resize = clip_logits.permute(0, 2, 1).contiguous().view(1, -1, patch_num, patch_num)
        return logit_resize


    def predict(self, inputs, data_samples, visualize=False):
        assert inputs.shape[0] == 1

        h, w = inputs.shape[-2:]
        image = inputs[0].permute(1, 2, 0)
        image = image.to(device="cuda", dtype=torch.float32)

        image_channel_first_float = inputs[0].to(device="cuda", dtype=torch.float32) / 255.0

        ori_shape = data_samples[0].ori_shape[:2]
        image_tensor_for_sam2 = torch.stack([image])
        image_tensor_for_sam2 = self.sam_transform(image_tensor_for_sam2)

        if self.resize_method == 'multi_resolution':
            clip_inputs = []
            if "city_scape" in self.config:
                clip_inputs.append(self.clip_preprocess(image_channel_first_float[:,:, :image_channel_first_float.shape[2]//2]))
                clip_inputs.append(self.clip_preprocess(image_channel_first_float[:,:, image_channel_first_float.shape[2]//2:]))

            else:
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
            self.points_per_w = self.crop_size // self.patch_size
            self.points_per_h = self.crop_size // self.patch_size
            clip_inputs = self.clip_preprocess(image_channel_first_float).unsqueeze(0)

        if self.clip_pretrained in ['openai', 'laion2b_s32b_b79k']:
            clip_inputs = clip_inputs.to(self.device, dtype=self.clip.visual.proj.dtype)
            if self.pool_feature == 'output':
                clip_last_attn_output = self.clip.encode_image(clip_inputs, return_last_attn_output=True)
            else:
                clip_last_blk_input, clip_last_blk = self.clip.encode_image(clip_inputs, return_value=True)

        elif self.clip_pretrained == 'siglip2':
            siglip2_inputs = clip_inputs.to(device=self.device, dtype=self.dtype)
            siglip_last_blk_input, intermediates = self.clip.visual.trunk.forward_intermediates(siglip2_inputs)
            siglip_last_blk = self.clip.visual.trunk.attn_pool
        elif self.clip_pretrained == 'meta':
            pe_inputs = clip_inputs.to(device=self.device, dtype=self.clip.visual.proj.dtype)
            pe_last_blk_input, pe_last_blk = self.clip.encode_image(pe_inputs, return_value=True, region_attn_mask=None)


        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
            sam2_masks = self.sam2_generator.generate_for_batch(image_tensor_for_sam2, [ori_shape], None)

        region_masks = torch.stack([mask['segmentations'] for mask in sam2_masks[0]])
        region_masks = region_masks.to(self.device, dtype=self.dtype)


        if self.upsample_times > 1:
            self.points_per_h *= self.upsample_times
            self.points_per_w *= self.upsample_times

        feature_masks = F.interpolate(region_masks.unsqueeze(0), [self.points_per_h, self.points_per_w], mode="bilinear")
        feature_masks = feature_masks.reshape(-1, self.points_per_h * self.points_per_w)

        if self.mask_type == 'soft':
            feature_masks = torch.clamp(feature_masks, min=0, max=1)
        elif self.mask_type == 'hard':
            feature_masks = (feature_masks > 0).float()
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")

        keep_masks = torch.sum(feature_masks, dim=1) > 0
        region_masks = region_masks[keep_masks]
        feature_masks = feature_masks[keep_masks]


        if self.pool_feature == 'value':
            if self.clip_pretrained in ['openai', 'laion2b_s32b_b79k']:
                region_logits, keep_masks = self.clip_value_with_sam2_attn(feature_masks, clip_last_blk_input, clip_last_blk, self.query_features)
            elif self.clip_pretrained == 'siglip2':
                region_logits, keep_masks = self.siglip_value_with_sam2_attn(feature_masks, siglip_last_blk_input, siglip_last_blk, self.query_features, self.logit_scale, self.logit_bias)
            elif self.clip_pretrained == 'meta':
                region_logits, keep_masks = self.pe_value_with_sam2_attn(feature_masks, pe_last_blk_input, pe_last_blk, self.query_features)
        elif self.pool_feature == 'input':
            region_logits, keep_masks = self.clip_feature_with_sam2(feature_masks, clip_last_blk_input, self.query_features)
        elif self.pool_feature == 'output':
            region_logits, keep_masks = self.clip_feature_with_sam2(feature_masks, clip_last_attn_output, self.query_features)

        region_masks = region_masks[keep_masks]
        region_logits = region_logits.to(dtype=self.dtype)
        clip_inputs_shape = clip_inputs.shape[-2:]



        if self.clip_pretrained in ['openai', 'laion2b_s32b_b79k'] and self.pool_feature != 'output':
            clip_logits = self.get_CLIP_logits(clip_last_blk_input, clip_last_blk, self.query_features)
        else:
            clip_logits = None


        seg_logits, seg_preds = self.postprocess_result(region_logits, region_masks, clip_inputs_shape, ori_shape, clip_logits)

        if data_samples is None:
            return seg_preds, seg_logits
        else:
            batch_size = seg_logits.shape[0]
            for i in range(batch_size):
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits[i]}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_preds})
                })
        return data_samples


    def postprocess_result(self, region_logits, region_masks, clip_inputs_shape, ori_shape, clip_logits=None, batch_size=100):

        if self.mask_type == 'soft':
            region_masks = torch.clamp(region_masks, min=0, max=1)
        elif self.mask_type == 'hard':
            region_masks = (region_masks > 0).to(dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")

        try:
            seg_logits = region_logits.unsqueeze(-1).unsqueeze(-1) * region_masks.unsqueeze(1)
            seg_logits = seg_logits.sum(0, keepdim=True)
        except:
            finish_compute_logits = False
            while finish_compute_logits is False:
                try:
                    seg_logits = torch.zeros(1, region_logits.size(1), region_masks.size(-2), region_masks.size(-1))
                    seg_logits = seg_logits.to(device=self.device, dtype=self.dtype)
                    for start in range(0, len(region_logits), batch_size):
                        end = start + batch_size
                        logits_batch = region_logits[start:end].unsqueeze(-1).unsqueeze(-1) * region_masks[start:end].unsqueeze(1)
                        seg_logits += logits_batch.sum(0, keepdim=True)
                    finish_compute_logits = True
                except:
                    batch_size = batch_size // 2
                    if batch_size == 0:
                        raise ValueError('Batch size is too small to process the data')

        if clip_logits is not None:
            null_pixel = torch.nonzero(region_masks.sum(0) == 0, as_tuple=True)
            clip_logits = nn.functional.interpolate(clip_logits, size=region_masks.shape[-2:], mode='bilinear')
            clip_logits = clip_logits.to(device=self.device, dtype=seg_logits.dtype)
            seg_logits[:, :, null_pixel[0], null_pixel[1]] = clip_logits[:, :, null_pixel[0], null_pixel[1]]

        seg_logits = F.interpolate(seg_logits, size=ori_shape, mode='bilinear')
        seg_logits = torch.softmax(seg_logits * self.region_logit_scale, dim=1)

        if self.num_cls != self.num_queries:
            seg_fg = seg_logits[:, -self.num_cls + 1:]
            seg_bg = seg_logits[:, :-self.num_cls + 1].max(1)[0]
            seg_logits = torch.cat([seg_bg.unsqueeze(0), seg_fg], dim=1)

        seg_preds = seg_logits.argmax(1)

        if self.num_cls != self.num_queries or self.prob_thd != 0:
            seg_preds[seg_logits.max(1)[0] < self.prob_thd] = 0
        return seg_logits, seg_preds



    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices