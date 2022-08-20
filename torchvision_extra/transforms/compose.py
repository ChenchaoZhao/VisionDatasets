from torchvision.transforms import Compose as _Compose

__all__ = ["Compose"]


class Compose(_Compose):
    """
    Compose.

    A better version of torchvision Compose.
    """

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)

        args = tuple(args)

        return args if len(args) > 1 else args[0]
