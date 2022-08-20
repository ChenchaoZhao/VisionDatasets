from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import PIL
from torchvision.datasets import OxfordIIITPet

__all__ = ["OxfordIIITPetPlus"]


class OxfordIIITPetPlus(OxfordIIITPet):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:
                - ``category`` (int): Label for one of the 37 pet categories.
                - ``coarse_category`` (int): Label for cat or dog
                - ``segmentation`` (PIL image): Segmentation trimap of the image.
                - ``detection``: Bounding box around segmentation mask
            If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):

        super().__init__(
            root,
            split,
            ("category", "segmentation"),
            transforms,
            transform,
            target_transform,
            download,
        )

        self.read_me = self._load_readme()
        # load species and breed labels
        self._load_labels()
        self._compute_boxes()
        self.target_types = target_types

    def _compute_boxes(self):
        self._boxes = []
        for idx in range(len(self)):
            mask = PIL.Image.open(self._segs[idx])
            ydx, xdx = np.where(np.array(mask) == 3)
            box = [[xdx.min(), ydx.min(), xdx.max(), ydx.max()]]
            self._boxes.append(box)

    def _load_readme(self) -> str:
        with open(self._anns_folder / "README", "r") as f:
            return f.read()

    def _load_labels(self) -> None:

        self.class_to_idx_tuple = {}
        self.idx_to_name = {}
        self.idx_to_species_breed_idx = {}

        with open(self._anns_folder / "list.txt", "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("#"):
                continue
            fn, general, coarse, fine = line.split(" ")

            name = " ".join((w.capitalize() for w in fn.lower().split("_")[:-1]))
            if name not in self.class_to_idx_tuple:
                if int(coarse) - 1 == 0:
                    coarse_name = "Cat"
                elif int(coarse) - 1 == 1:
                    coarse_name = "Dog"
                else:
                    raise ValueError(
                        f"Coarse label should only be 1 or 2 but got {coarse}"
                    )

                name = f"{name}, {coarse_name}"

                self.class_to_idx_tuple[name] = (
                    int(general) - 1,
                    int(coarse) - 1,
                    int(fine) - 1,
                )
                self.idx_to_name[int(general) - 1] = name
                self.idx_to_species_breed_idx[int(general) - 1] = (
                    int(coarse) - 1,
                    int(fine) - 1,
                )

        self.species_to_species_idx = {"Cat": 0, "Dog": 1}
        self.cat_breed_to_breed_idx = {}
        self.dog_breed_to_breed_idx = {}

        for name, idx_tuple in self.class_to_idx_tuple.items():
            breed, species = name.split(", ")
            if species == "Cat":
                self.cat_breed_to_breed_idx[breed] = idx_tuple[-1]
            else:  # species == 'Dog'
                self.dog_breed_to_breed_idx[breed] = idx_tuple[-1]

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        image = PIL.Image.open(self._images[idx]).convert("RGB")

        target: Dict[str, Any] = {}
        for target_type in self._target_types:
            if target_type == "category":
                target["flat_label"] = self._labels[idx]
            elif target_type == "coarse_category":
                species_label, breed_label = self.idx_to_species_breed_idx[
                    self._labels[idx]
                ]
                target["flat_label"] = self._labels[idx]
                target["species_label"] = species_label
                target["breed_label"] = breed_label
            elif target_type == "detection":
                target["boxes"] = self._boxes[idx]
            elif target_type == "segmentation":
                target.append(PIL.Image.open(self._segs[idx]))
            else:
                raise KeyError(f"Target type ({target_type}) not found.")

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
