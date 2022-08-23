import warnings
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
        has_cat: bool = True,
        has_dog: bool = True,
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
        self.cat_subset_indices = []
        self.dog_subset_indices = []
        for idx, l in enumerate(self._labels):
            if l in self.cat_flat_labels:
                self.cat_subset_indices.append(idx)
            else:
                self.dog_subset_indices.append(idx)
        self.target_types = target_types
        self.has_cat = has_cat
        self.has_dog = has_dog
        self.cat_classes = tuple(self.cat_breed_to_breed_idx.keys())
        self.dog_classes = tuple(self.dog_breed_to_breed_idx)

    def _compute_boxes(self):
        self._boxes = []
        for idx in range(len(self._labels)):
            mask = PIL.Image.open(self._segs[idx])
            try:
                ydx, xdx = np.where(np.array(mask) == 3)
                box = [[xdx.min(), ydx.min(), xdx.max(), ydx.max()]]
            except ValueError:
                warnings.warn(f"Data item {idx} does not have a valid mask or bbox.")
                W, H = mask.size
                box = [[0.0, 0.0, W, H]]
            self._boxes.append(box)

    def _load_readme(self) -> str:
        with open(self._anns_folder / "README", "r") as f:
            return f.read()

    def _load_labels(self) -> None:

        self.class_to_idx_tuple = {}
        self.idx_to_name = {}
        self.idx_to_species_breed_idx = {}
        self.cat_flat_labels = set()
        self.dog_flat_labels = set()

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
                    self.cat_flat_labels.add(int(general) - 1)
                elif int(coarse) - 1 == 1:
                    coarse_name = "Dog"
                    self.dog_flat_labels.add(int(general) - 1)
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
        if self.has_cat and self.has_dog:
            if idx < 0:
                idx = len(self) + idx
                assert idx >= 0
            return self._get_item(idx)

        if self.has_cat and not self.has_dog:
            # cat only
            return self._get_item(self.cat_subset_indices[idx])

        if not self.has_cat and self.has_dog:
            # dog only
            return self._get_item(self.dog_subset_indices[idx])

    def _get_item(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        image = PIL.Image.open(self._images[idx]).convert("RGB")

        target: Dict[str, Any] = {}
        for target_type in self.target_types:
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
                target["segmentation"] = PIL.Image.open(self._segs[idx])
            else:
                raise KeyError(f"Target type ({target_type}) not found.")

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:

        if self.has_cat and self.has_dog:
            return super().__len__()

        if self.has_cat and not self.has_dog:
            # cat only
            return len(self.cat_subset_indices)

        if not self.has_cat and self.has_dog:
            # dog only
            return len(self.dog_subset_indices)

        return 0
