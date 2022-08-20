from typing import Any, Callable, Dict, Optional, Tuple

from PIL.Image import Image as PilImage

__all__ = ["DictionaryTransforms", "DictionaryTransform"]


class _DictionaryTransformBase(Callable):
    """
    _DictionaryTransformBase.

    A wrapper of Albumentations-style keyword args transforms. Compatible with
    torchvision transforms.
    """

    def __init__(
        self,
        preprocessing: Callable[[PilImage, Any], Dict[str, Any]],
        dictionary_transform: Callable[..., Dict[str, Any]],
        postprocessing: Callable[
            [Dict[str, Any]], Tuple[PilImage, Optional[Dict[str, Any]]]
        ],
    ):
        """
        __init__

        Args:
            preprocessing (Callable[[PilImage, Any], Dict[str, Any]]): Convert any target type into a dict
            dictionary_transform (Callable[..., Dict[str, Any]]): Only kwargs as inputs and then outputs a dict
            postprocessing (Callable[ [Dict[str, Any]], Tuple[PilImage, Optional[Dict[str, Any]]] ]):
                Converts dict outputs to torchvision (image, target) style
        """

        self.preprocessing = preprocessing
        self.core = dictionary_transform
        self.postprocessing = postprocessing

    def __call__(self, *args):
        out = self.preprocessing(*args)
        out = self.core(**out)  # always outputs an dict
        out = self.postprocessing(out)
        return out

    def __repr__(self):
        return repr(self.core)

    __str__ = __repr__


class DictionaryTransform(_DictionaryTransformBase):
    def __call__(self, image: PilImage) -> Any:
        return super().__call__(image)


class DictionaryTransforms(_DictionaryTransformBase):
    def __call__(self, image: PilImage, target: Any) -> Tuple[Any, Any]:
        return super().__call__(image, target)
