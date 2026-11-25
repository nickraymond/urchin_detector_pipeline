"""S3 bite 4: the overlap rules of D22 on a hand-made sam_raw record (no SAM, no images)."""

import numpy as np

from app.sam import classes, overlap, results

W, H = 100, 60
CLASSES = classes.parse(classes.DEFAULT_FILE.read_text())["classes"]
# ids: 1 purple urchin, 2 red urchin (group urchin), 3 sea star, 8 kelp shoot (seg_as kelp),
# 9 kelp (area, listed before rock), 7 rock/substrate (area)


def mask(box, score=0.9, class_id=None, prompt="p"):
    x0, y0, x1, y1 = box
    det = {"box": list(box), "mask": np.ones((y1 - y0, x1 - x0), bool)}
    return {"prompt": prompt, "score": score, "box": list(box), "area": (x1 - x0) * (y1 - y0),
            "rle": results.encode(det, W, H), "class_id": class_id}


def raw(**units):
    return {"width": W, "height": H, "units": {k: {"masks": v} for k, v in units.items()}}


def test_areas_first_listed_wins_prompts_union_and_unknown_is_zero():
    r = raw(**{"rock/substrate": [mask((0, 0, 90, 60), class_id=7)],
               "kelp": [mask((0, 0, 30, 60), class_id=9), mask((20, 0, 50, 60), class_id=9)]})
    m = overlap.class_map(r, CLASSES)
    assert m.dtype == np.uint8 and m.shape == (H, W)
    assert (m[:, :50] == 9).all()          # kelp over rock (rule 3); two prompts unioned (rule 2)
    assert (m[:, 50:90] == 7).all() and (m[:, 90:] == overlap.UNKNOWN).all()   # rule 6


def test_iou_clash_keeps_higher_score_and_flags_it():
    r = raw(urchin=[mask((10, 10, 30, 30), 0.9)], **{"sea star": [mask((11, 10, 30, 30), 0.8,
                                                                        class_id=3)]})
    objs = overlap.objects(r, CLASSES)
    assert [(o["unit"], o["flag"]) for o in objs] == [("urchin", True)]         # rule 1
    assert (overlap.class_map(r, CLASSES)[10:30, 10:30] == overlap.UNTAGGED).all()  # rule 7


def test_objects_on_areas_higher_score_takes_shared_pixels_and_seg_as():
    r = raw(**{"rock/substrate": [mask((0, 0, W, H), class_id=7)],
               "urchin": [mask((40, 40, 50, 50), 0.6, class_id=1)],
               "sea star": [mask((45, 45, 60, 55), 0.7, class_id=3)],
               "kelp shoot": [mask((70, 10, 80, 20), 0.5, class_id=8)]})
    m = overlap.class_map(r, CLASSES)
    assert m[42, 42] == 1 and m[47, 47] == 3 and m[50, 58] == 3     # rule 4, no IoU clash
    assert not any(o["flag"] for o in overlap.objects(r, CLASSES))
    assert (m[10:20, 70:80] == 9).all()                             # shoot paints kelp (rule 5)
    assert m[5, 5] == 7                                             # objects on top of areas


def test_classes_not_in_the_file_are_left_out():
    r = raw(**{"old class": [mask((0, 0, 10, 10), class_id=42)],
               "sea star": [mask((0, 0, 10, 10), class_id=42)]})   # tag of a removed class
    assert overlap.objects(r, CLASSES) == []
    assert (overlap.class_map(r, CLASSES) == 0).all()
