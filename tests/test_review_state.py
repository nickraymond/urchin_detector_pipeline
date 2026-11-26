"""S5 bite 1: stable mask ids, review state in state.sqlite, final masks after review, accept
rule and undo (D40, D41, D18, D31). Synthetic masks only: no SAM, no images."""

import json

import numpy as np
import pytest

from app import db
from app.review import state
from app.sam import classes, overlap, results

W, H = 100, 60
PARSED = classes.parse(classes.DEFAULT_FILE.read_text())
CLASSES, SETTINGS = PARSED["classes"], PARSED["settings"]
UNITS = {u["key"]: u for u in classes.run_units(CLASSES)}
# ids: 1 purple urchin, 2 red urchin (group urchin), 3 sea star, 6 fish, 9 kelp (area, listed
# before rock), 7 rock/substrate (area)
F = "f.jpg"


def det(box, score=0.9, prompt="p"):
    x0, y0, x1, y1 = box
    return {"prompt": prompt, "score": score, "box": list(box),
            "mask": np.ones((y1 - y0, x1 - x0), bool)}


def run(folder, unit, dets, settings=SETTINGS):
    """Save one unit's results the way the SAM worker does, keeping reviewed masks."""
    return results.write_frame(folder, F, W, H, settings,
                               {unit: {"unit": UNITS[unit], "seconds": 0.1, "dets": dets}},
                               state.reviewed_ids(folder, F))


@pytest.fixture
def proj(tmp_path):
    conn = db.connect(tmp_path / "state.sqlite", create=True)
    run(tmp_path, "urchin", [det((10, 10, 20, 20), 0.9), det((40, 10, 50, 20), 0.8)])
    run(tmp_path, "sea star", [det((60, 30, 70, 40), 0.7)])
    run(tmp_path, "kelp", [det((0, 40, 50, 60))])
    run(tmp_path, "rock/substrate", [det((0, 30, 100, 60))])
    yield tmp_path, conn
    conn.close()


def ids(fin):
    return sorted(o["mask_id"] for o in fin["objects"])


def test_mask_ids_are_stable_and_never_reused(proj):
    folder, conn = proj
    raw = results.read(folder, F)
    assert [m["mask_id"] for m in raw["units"]["urchin"]["masks"]] == [1, 2]
    assert raw["units"]["sea star"]["masks"][0]["mask_id"] == 3 and raw["next_mask_id"] == 6
    state.set_class(conn, folder, F, [1], 1, CLASSES)            # mask 1 reviewed: purple
    data = run(folder, "urchin", [det((80, 0, 90, 10), 0.95)], {**SETTINGS, "confidence": 0.5})
    masks = data["units"]["urchin"]["masks"]
    assert [(m["mask_id"], m["from_older_settings"]) for m in masks] == [(1, True), (6, False)]
    assert data["next_mask_id"] == 7                             # id 2 (replaced) is not reused


def test_a_file_from_before_s5_gets_the_same_ids_every_read(tmp_path):
    run(tmp_path, "urchin", [det((10, 10, 20, 20)), det((40, 10, 50, 20))])
    p = results.raw_path(tmp_path, F)
    old = json.loads(p.read_text())
    for m in old["units"]["urchin"]["masks"]:
        del m["mask_id"]
    del old["next_mask_id"]
    p.write_text(json.dumps(old))
    first, second = results.read(tmp_path, F), results.read(tmp_path, F)
    assert [m["mask_id"] for m in first["units"]["urchin"]["masks"]] == [1, 2]
    assert first == second and "mask_id" not in p.read_text()   # reading never writes


def test_final_masks_tagging_delete_and_class_map(proj):
    folder, conn = proj
    fin = state.final(folder, conn, F, CLASSES)
    assert ids(fin) == [1, 2, 3] and fin["untagged"] == [1, 2]
    assert sorted(a["class_id"] for a in fin["areas"]) == [7, 9]
    state.set_class(conn, folder, F, [1], 2, CLASSES)            # tag red
    state.delete(conn, folder, F, [3])                           # X the sea star
    fin = state.final(folder, conn, F, CLASSES)
    assert ids(fin) == [1, 2] and fin["untagged"] == [2]
    cmap = state.class_map(fin, CLASSES)
    assert (cmap[10:20, 10:20] == 2).all() and (cmap[10:20, 40:50] == overlap.UNTAGGED).all()
    assert (cmap[30:40, 60:70] == 7).all()                       # the deleted star shows rock
    assert (cmap[45:60, 0:50] == 9).all() and (cmap[0:10, 60:] == 0).all()
    assert results.read(folder, F)["units"]["urchin"]["masks"][0]["class_id"] is None  # raw kept


def test_wrong_kind_unknown_class_and_unknown_mask_are_refused(proj):
    folder, conn = proj
    kelp_id = results.read(folder, F)["units"]["kelp"]["masks"][0]["mask_id"]
    with pytest.raises(state.ReviewError, match="same kind"):
        state.set_class(conn, folder, F, [1], 9, CLASSES)        # urchin -> kelp (area)
    with pytest.raises(state.ReviewError, match="same kind"):
        state.set_class(conn, folder, F, [kelp_id], 1, CLASSES)
    with pytest.raises(state.ReviewError, match="No class"):
        state.set_class(conn, folder, F, [1], 99, CLASSES)
    with pytest.raises(state.ReviewError, match="No mask"):
        state.delete(conn, folder, F, [42])
    assert state.mask_rows(conn, F) == {}                        # nothing half-done


def test_an_area_mask_moved_to_another_area_class_paints_in_that_class_place(proj):
    folder, conn = proj
    kelp_id = results.read(folder, F)["units"]["kelp"]["masks"][0]["mask_id"]
    state.set_class(conn, folder, F, [kelp_id], 7, CLASSES)
    cmap = state.class_map(state.final(folder, conn, F, CLASSES), CLASSES)
    assert (cmap[45:60, 0:50] == 7).all() and not (cmap == 9).any()


def test_accept_is_refused_with_an_untagged_urchin_and_marks_masks_reviewed(proj):
    folder, conn = proj
    with pytest.raises(state.ReviewError, match="2 urchins on this frame have no type"):
        state.accept(conn, folder, F, CLASSES)
    assert not state.frame_state(conn, F)["accepted"]
    state.set_class(conn, folder, F, [1, 2], 1, CLASSES)
    state.accept(conn, folder, F, CLASSES)
    assert state.frame_state(conn, F)["accepted"]
    assert state.reviewed_ids(folder, F) == {1, 2, 3, 4, 5}      # objects and areas


def test_a_deleted_mask_stays_hidden_after_a_re_run(proj):
    """D31 extended: the re-run's fresh copy of a mask the reviewer deleted is not shown."""
    folder, conn = proj
    state.delete(conn, folder, F, [3])
    run(folder, "sea star", [det((60, 30, 70, 40), 0.99)], {**SETTINGS, "confidence": 0.5})
    assert ids(state.final(folder, conn, F, CLASSES)) == [1, 2]
    # among reviewed masks, a kept one beats a deleted one with a higher score
    state.set_class(conn, folder, F, [6], 3, CLASSES)            # the fresh star, reviewed
    assert 6 in ids(state.final(folder, conn, F, CLASSES))


def test_masks_added_by_hand_are_objects_of_their_class(proj):
    folder, conn = proj
    p = results.raw_path(folder, F)
    raw = json.loads(p.read_text())
    d = det((85, 45, 95, 55), 1.0)
    raw["units"][overlap.ADDED] = {"masks": [{
        "mask_id": raw["next_mask_id"], "prompt": "click", "score": 1.0, "box": d["box"],
        "area": 100, "rle": results.encode(d, W, H), "class_id": 6, "added_by_hand": True}]}
    p.write_text(json.dumps(raw))
    fin = state.final(folder, conn, F, CLASSES)
    added = [o for o in fin["objects"] if o["added"]]
    assert [(o["mask_id"], o["class_id"]) for o in added] == [(6, 6)]
    assert (state.class_map(fin, CLASSES)[45:55, 85:95] == 6).all()


def test_split_check_fully_checked_and_undo_of_every_action(proj):
    folder, conn = proj
    assert state.frame_state(conn, F) == {"accepted": False, "split_ok": None, "checked": []}
    state.set_class(conn, folder, F, [1, 2], 1, CLASSES)
    state.delete(conn, folder, F, [3])
    state.accept(conn, folder, F, CLASSES)
    state.set_split(conn, F, False)
    state.set_checked(conn, F, 1, True, CLASSES)
    state.set_checked(conn, F, 3, True, CLASSES)
    state.reopen(conn, F)
    assert state.frame_state(conn, F) == {"accepted": False, "split_ok": False, "checked": [1, 3]}
    assert state.undo(conn, "other.jpg") is None                 # per frame
    steps = [state.undo(conn, F) for _ in range(7)]
    assert [s.split(" on ")[0] for s in steps] == [
        "reopen", "sea star checked", "purple urchin checked", "split check no", "accept",
        "delete", "class purple urchin"]
    assert state.undo(conn) is None
    assert state.mask_rows(conn, F) == {}
    assert state.frame_state(conn, F) == {"accepted": False, "split_ok": None, "checked": []}
    assert ids(state.final(folder, conn, F, CLASSES)) == [1, 2, 3]


def test_undo_puts_back_the_previous_review_not_a_blank(proj):
    folder, conn = proj
    state.set_class(conn, folder, F, [1], 2, CLASSES)
    state.set_class(conn, folder, F, [1], 1, CLASSES)
    state.undo(conn, F)
    assert state.mask_rows(conn, F)[1]["class_id"] == 2


def test_reviewed_ids_without_a_database_or_before_s5(tmp_path):
    assert state.reviewed_ids(tmp_path, F) == set()
    import sqlite3
    sqlite3.connect(tmp_path / "state.sqlite").close()           # an old file, no tables
    assert state.reviewed_ids(tmp_path, F) == set()
