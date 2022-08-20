from typing import Sequence, Union

__all__ = ["Vocabulary"]


class Vocabulary:
    """
    Vocabulary.

    Use `__getitem__` operator to fetch class name or class index
    """

    def __init__(self, name: str, class_names: Sequence[str]):
        """
        __init__

        Args:
            name (str): Vocabulary name
            class_names (Sequence[str]): A sequence of class names
        """
        self.name = str(name)
        self.class_names = tuple([str(n) for n in class_names])
        self.name_to_idx = {c: i for i, c in enumerate(self.class_names)}

    def __getitem__(self, key_or_idx: Union[int, str]):
        if isinstance(key_or_idx, str):
            return self.name_to_idx[key_or_idx]
        elif isinstance(key_or_idx, int):
            return self.class_names[key_or_idx]
        else:
            raise TypeError(f"Key can be `int` or `str` but got `{type(key_or_idx)}`")

    def __len__(self):
        return len(self.class_names)

    def __repr__(self):
        head = self.__class__.__name__
        body = []
        body.append("(")
        body.append(f"name={repr(self.name)}")
        body.append(", ")
        body.append(f"class_names={repr(self.class_names)}")
        body.append(")")
        return head + "".join(body)

    __str__ = __repr__
