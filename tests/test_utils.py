import torchvision_extra as vx


def test_vocabulary():
    vocab = vx.utils.Vocabulary("test", ["a", "b", "c"])
    assert len(vocab) == 3
    assert vocab["a"] == 0
    assert vocab["b"] == 1
    assert vocab["c"] == 2
    assert vocab[0] == "a"
    assert vocab[1] == "b"
    assert vocab[2] == "c"
