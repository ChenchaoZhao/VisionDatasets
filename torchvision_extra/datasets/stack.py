import random
from typing import *

from torch.utils.data import Dataset

__all__ = ["StackDataset"]


class StackDataset(Dataset):

    _repr_indent = 4

    def __init__(self, datasets: Dict[str, Dataset], pad_short: bool = True):
        super().__init__()
        self.pad_short = pad_short
        assert isinstance(
            datasets, dict
        ), f"Input `datasets` should be a dict but got type {type(datasets)}"

        self.datasets = datasets
        max_len = 0
        min_len = float("inf")
        data_len = {}
        for name, ds in datasets.items():
            max_len = max(len(ds), max_len)
            min_len = min(len(ds), min_len)
            data_len[name] = len(ds)
        self.min_len = max(0, min_len)
        self.max_len = max(self.min_len, max_len)
        self.data_len = data_len
        self._num_output_args = {
            vocab: len(self.datasets[vocab][0]) for vocab in self.datasets
        }

    def __getitem__(self, idx):

        if idx < 0:
            jdx = len(self) + idx
            if jdx < 0:
                raise IndexError(
                    f"Index {idx} is out of bound [{-len(self)}, {len(self)})."
                )
            idx = jdx
        if idx >= len(self):
            raise IndexError(f"Index {idx} >= {len(self)} is out of bound.")

        outputs = {}

        for name, ds in self.datasets.items():
            if idx >= self.data_len[name]:
                jdx = random.randint(0, self.data_len[name] - 1)
            else:
                jdx = idx
            outputs[name] = ds[jdx]

        return outputs

    def __len__(self) -> int:
        if self.pad_short:
            return self.max_len
        else:
            return self.min_len

    def __repr__(self) -> str:
        out = [
            "StackDataset",
            " " * self._repr_indent + f"`pad_short`: {self.pad_short}",
            " " * self._repr_indent + f"`len`: {len(self)}",
        ]
        for name, ds in self.datasets.items():
            out.append(f"`{name}`:")
            r = repr(ds)
            for l in r.split("\n"):
                out.append(" " * self._repr_indent + f"{l}")

        return "\n".join(out)

    def collate_fn(
        self, batch: List[Dict[str, Tuple[Any]]]
    ) -> Dict[str, Tuple[List[Any]]]:
        """
        `collate_fn` for `StackDataset` items.

        Args:
            batch (List[Dict[str, Tuple[Any]]]): A list of dict items, i.e. vocab name -> data item

        Returns:
            Dict[str, Tuple[List[Any]]]: A dict of batched items, i.e. vocab name -> tuple of args where args are List[Any]
        """

        batched = {
            vocab: [[] for _ in range(n)] for vocab, n in self._num_output_args.items()
        }

        for item in batch:
            for vocab, n in self._num_output_args.items():
                for adx in range(n):
                    batched[vocab][adx].append(item[vocab][adx])

        return batched
