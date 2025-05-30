# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from sam2.utils.transforms import SAM2Transforms

import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, ToTensor
import torch.nn.functional as F
import torchvision.transforms as transforms



class CustomSAM2Transforms(SAM2Transforms):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )
        self.batched_transform = torch.vmap(self.transforms)

        self.batch_mean = torch.tensor(self.mean).view(1, -1, 1, 1)
        self.batch_std = torch.tensor(self.std).view(1, -1, 1, 1)


    def __call__(self, x, device=None, dtype=torch.bfloat16):
        if type(x) is not torch.Tensor:
            x = self.to_tensor(x)
        if device is not None:
            x = x.to(device=device, dtype=dtype)

        if len(x.shape) == 3:
            x = self.transforms(x)
        elif len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2) / 255.0
            assert x.shape[1] == 3, f"Expected 3 channels, got x with shape {x.shape}"
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
            x = (x - self.batch_mean.to(x.device)) / self.batch_std.to(x.device)
            # x = self.batched_transform(x)
        return x
