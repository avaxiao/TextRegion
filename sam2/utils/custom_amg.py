
import torch
import numpy as np
from sam2.utils.amg import MaskData



class CustomMaskData(MaskData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                # self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
                v = v[keep]
                self._stats[k] = v
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")
