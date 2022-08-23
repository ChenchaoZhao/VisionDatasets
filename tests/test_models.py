import torchvision_extra as vx


def test_faster_rcnn_head():
    head = vx.models.detection.faster_rcnn.MultiVocabularyFastRCNNPredictor(
        8, dict(a=3, b=5, c=8)
    )

    head.set_vocabulary("a")
    assert head.label_shift == 1
    a = head.export_current_vocab_heads()
    assert a.cls_score.in_features == 8
    assert a.cls_score.out_features == 4
    assert a.bbox_pred.in_features == 8
    assert a.bbox_pred.out_features == 4 * 4

    head.set_vocabulary("b")
    assert head.label_shift == 1 + 3
    b = head.export_current_vocab_heads()
    assert b.cls_score.in_features == 8
    assert b.cls_score.out_features == 6
    assert b.bbox_pred.in_features == 8
    assert b.bbox_pred.out_features == 4 * 6

    head.set_vocabulary("c")
    assert head.label_shift == 1 + 3 + 5
    c = head.export_current_vocab_heads()
    assert c.cls_score.in_features == 8
    assert c.cls_score.out_features == 9
    assert c.bbox_pred.in_features == 8
    assert c.bbox_pred.out_features == 4 * 9
