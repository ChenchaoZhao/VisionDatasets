import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

__all__ = ["MultiVocabularyFastRCNNPredictor"]


class MultiVocabularyFastRCNNPredictor(FastRCNNPredictor):
    """
    MultiVocabularyFastRCNNPredictor.

    Train a FasterRCNN using multiple Vocabularies Set vocabulary using
    `set_vocabulary` Export partial head for a single vocabulary using
    `export_current_vocab_heads`
    """

    def __init__(self, in_channels, vocabulary_sizes: Dict[str, int]):
        vocabulary_sizes = {str(k): int(v) for k, v in vocabulary_sizes.items()}
        torch._assert(
            all([v > 0 for v in vocabulary_sizes.values()]),
            f"Vocabulary size should be more than one but got {vocabulary_sizes}",
        )
        total_num_classes = 1 + sum((v for v in vocabulary_sizes.values()))
        # __background__ is a default class with index 0
        super().__init__(in_channels, total_num_classes)
        self.total_num_classes = total_num_classes
        self._vocabulary_sizes = vocabulary_sizes
        vocab_to_range = {}  # '__background__' is 0
        current_index = 0
        for vocab, size in vocabulary_sizes.items():
            vocab_to_range[vocab] = (current_index + 1, current_index + size)
            current_index += size
        self._vocabulary_to_range = vocab_to_range
        self.__current_vocabulary = "all"

    def extra_repr(self) -> str:
        return f"current_vocabulary: {self.current_vocabulary}"

    def _is_in_vocab(self, label: int, vocab_name: str) -> bool:
        min_idx, max_idx = self._vocabulary_to_range[vocab_name]
        return min_idx <= label <= max_idx

    def set_vocabulary(self, vocab_name: str = "all") -> None:
        if vocab_name != "all":
            torch._assert(
                vocab_name in self._vocabulary_sizes,
                f"Vocabulary `{vocab_name}` not found.",
            )
        self.__current_vocabulary = vocab_name

    @property
    def label_shift(self) -> int:
        return self._vocabulary_to_range[self.current_vocabulary][0]

    @property
    def current_vocabulary(self) -> str:
        return self.__current_vocabulary

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        logits, box_deltas = super().forward(x)
        if self.current_vocabulary == "all":
            return logits, box_deltas

        min_idx, max_idx = self._vocabulary_to_range[self.current_vocabulary]
        logits[:, 1:min_idx] += float("-inf")
        logits[:, max_idx + 1 :] += float("-inf")

        return logits, box_deltas

    def export_current_vocab_heads(self) -> FastRCNNPredictor:

        in_channels = self.cls_score.in_features
        cls_w = copy.deepcopy(self.cls_score.weight).cpu()
        cls_b = copy.deepcopy(self.cls_score.bias).cpu()
        box_w = copy.deepcopy(self.bbox_pred.weight).cpu()
        box_b = copy.deepcopy(self.bbox_pred.bias).cpu()

        if self.current_vocabulary == "all":
            m = FastRCNNPredictor(in_channels, self.total_num_classes)
        else:
            min_idx, max_idx = self._vocabulary_to_range[self.current_vocabulary]
            m = FastRCNNPredictor(
                in_channels, 1 + self._vocabulary_sizes[self.current_vocabulary]
            )

            cls_w = cls_w[[0] + [i for i in range(min_idx, max_idx + 1)]]
            cls_b = cls_b[[0] + [i for i in range(min_idx, max_idx + 1)]]
            box_w = box_w[
                [i for i in range(4)] + [i for i in range(4 * min_idx, 4 * max_idx + 1)]
            ]
            box_b = box_b[
                [i for i in range(4)] + [i for i in range(4 * min_idx, 4 * max_idx + 1)]
            ]

        m.cls_score.weight = nn.Parameter(cls_w)
        m.cls_score.bias = nn.Parameter(cls_b)
        m.bbox_pred.weight = nn.Parameter(box_w)
        m.bbox_pred.bias = nn.Parameter(box_b)

        return m
