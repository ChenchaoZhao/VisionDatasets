import os
import os.path
import pathlib
from typing import Callable, List, Optional, Tuple

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class StanfordDogs(VisionDataset):

    OUTPUT_FIELDS: Tuple[str] = ("image", "label", "bbox")

    _RESOURCES = (
        (
            "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            None,
        ),
        (
            "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
            None,
        ),
        (
            "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
            None,
        ),
    )

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
        self._image_folder = self._base_folder / "Images"
        self._anno_folder = self._base_folder / "Annotation"
        self._list_folder = self._base_folder / "lists"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self._images = []  # list of relative paths to images
        self._labels = []  # list of int labels (zero-based)
        self._annos = []  # list of relative paths to annotations
        self._load_split()  # populates _images, _labels, _annos

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(  # sort the set of (str, int) using int
                {
                    (
                        _image.split("/")[0].split("-", 1)[1],
                        label,
                    )  # split '-' from left up to 1 '-'
                    for _image, label in zip(self._images, self._labels)
                },
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        # e.g. n02085620-Chihuahua/n02085620_2650.jpg,
        #      n02095314-wire-haired_fox_terrier/n02095314_3052.jpg

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._image_folder / i for i in self._images]
        self._annos = [self._anno_folder / a for a in self._annos]

        assert len(self._images) == len(
            self._annos
        ), f"Number of images ({len(self._images)}) is not consistent with number of annotations ({len(self._annos)})"

        self._boxes = (
            []
        )  # list of List[List[int]], e.g. [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]
        self._load_boxes()

    def __len__(self) -> int:
        return len(self._images)

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(
                url, download_root=str(self._base_folder), md5=md5
            )

    def _check_exists(self) -> bool:
        for folder in (self._image_folder, self._anno_folder, self._list_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _load_split(self):
        import scipy.io

        split_info = scipy.io.loadmat(self._list_folder / f"{self._split}_list.mat")
        self._images = [f[0][0] for f in split_info["file_list"]]
        self._labels = [l[0] - 1 for l in split_info["labels"]]
        self._annos = [a[0][0] for a in split_info["annotation_list"]]

    def __load_boxes_per_image(self, path: str) -> List[int]:
        import xml.etree.ElementTree

        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter("object"):
            boxes.append(
                [
                    int(objs.find("bndbox").find("xmin").text),
                    int(objs.find("bndbox").find("ymin").text),
                    int(objs.find("bndbox").find("xmax").text),
                    int(objs.find("bndbox").find("ymax").text),
                ]
            )
        return boxes

    def _load_boxes(self):
        for anno in self._annos:
            self.boxes.append(self.__load_boxes_per_image(anno))
