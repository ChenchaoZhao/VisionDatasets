from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image as PilImage

__all__ = ["ToTensors"]


class ToTensors(Callable):
    """
    ToTensors.

    A better version of torchvision `ToTensor` which takes only one PIL image
    argument. `ToTensors` can convert any input types to torch.Tensor
    """

    def __init__(self, vocabulary: Optional[Dict[str, int]] = None):

        self.vocabulary = vocabulary

    def __call__(self, *args) -> Tuple[Any]:
        out = tuple([self._to_tensor(a) for a in args])
        return out if len(out) > 1 else out[0]

    def _to_tensor(self, x: Any):

        if isinstance(x, np.ndarray):
            return torch.tensor(x)

        if isinstance(x, (list, tuple, set)):

            as_list = [self._to_tensor(y) for y in x]
            # the elements are the same type `int` or `float`
            if all([isinstance(y, int) for y in as_list]):
                return torch.tensor(as_list).long()
            if all([isinstance(y, float) for y in as_list]):
                return torch.tensor(as_list).float()

            if all([isinstance(y, (list, tuple)) for y in x]):
                _dtype_size = None
                for item in as_list:
                    if not isinstance(item, torch.Tensor):
                        break
                    if _dtype_size:
                        _dt, _sz = _dtype_size
                        if _dt == item.dtype and _sz == item.size():
                            continue
                        else:
                            break
                    else:
                        _dtype_size = item.dtype, item.size()
                else:
                    return torch.stack(as_list, dim=0)

            # otherwise return tensors in the original container type
            return type(x)(as_list)

        if isinstance(x, dict):
            return {k: self._to_tensor(v) for k, v in x.items()}

        if isinstance(x, str):
            return self.vocabulary[x]

        if isinstance(x, (int, float)):
            return x

        if isinstance(x, PilImage):
            return TF.to_tensor(x)

        return x

    def __repr__(self) -> str:
        head = self.__class__.__name__
        body = f"(vocabulary={self.vocabulary})"
        return head + body
