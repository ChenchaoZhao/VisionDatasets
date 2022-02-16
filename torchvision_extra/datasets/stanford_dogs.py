import pathlib
from typing import Callable, Optional, Tuple

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset


class StanfordDogs(VisionDataset):

    output_fields: Tuple[str] = ("image", "label", "bbox")

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):

        self._split = verify_str_arg(split, "split", ("train", "test"))

        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self._base_folder = pathlib.Path(self.root) / "stanford-dogs"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "Annotation"
