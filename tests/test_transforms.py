import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

import torchvision_extra as vx


def test_to_tensors():

    transforms = vx.transforms.ToTensors()
    pil_image = PIL.Image.fromarray(np.random.randn(10, 12, 3), mode="RGB")
    assert transforms(pil_image).allclose(TF.to_tensor(pil_image))

    out = transforms(pil_image, pil_image)
    assert len(out) == 2
    assert out[0].allclose(TF.to_tensor(pil_image))
    assert out[1].allclose(TF.to_tensor(pil_image))

    out = transforms([1, 2, 3])
    assert isinstance(out, torch.LongTensor)
    assert (out == torch.tensor([1, 2, 3]).long()).all()

    out = transforms([[1], [2], [3]])
    assert isinstance(out, torch.LongTensor)
    assert (out == torch.tensor([[1], [2], [3]]).long()).all()

    out = transforms([[1, 2, 3]])
    assert isinstance(out, torch.LongTensor)
    assert (out == torch.tensor([[1, 2, 3]]).long()).all()

    out = transforms([0.1, 0.2, 0.3])
    assert isinstance(out, torch.FloatTensor)
    assert (out == torch.tensor([0.1, 0.2, 0.3]).float()).all()

    out = transforms([[0.1], [0.2], [0.3]])
    assert isinstance(out, torch.FloatTensor)
    assert (out == torch.tensor([[0.1], [0.2], [0.3]]).float()).all()

    out = transforms([[0.1, 0.2, 0.3]])
    assert isinstance(out, torch.FloatTensor)
    assert (out == torch.tensor([[0.1, 0.2, 0.3]]).float()).all()

    out = transforms({"labels": [1], "boxes": [[0.1, 0.2, 0.3, 0.4]]})
    assert (out["labels"] == torch.tensor([1]).long()).all()
    assert (out["boxes"] == torch.tensor([[0.1, 0.2, 0.3, 0.4]]).float()).all()

    img, out = transforms(pil_image, {"labels": [1], "boxes": [[0.1, 0.2, 0.3, 0.4]]})
    assert img.allclose(TF.to_tensor(pil_image))
    assert (out["labels"] == torch.tensor([1]).long()).all()
    assert (out["boxes"] == torch.tensor([[0.1, 0.2, 0.3, 0.4]]).float()).all()
