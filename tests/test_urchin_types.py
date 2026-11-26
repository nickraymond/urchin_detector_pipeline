"""S5 bite 4a: black urchin (D39), renames that keep the id, urchin types and fast tagging.
Synthetic masks only."""

import json

import numpy as np
import pytest

from app import db
from app.review import state, tagging
from app.sam import classes, results

DEFAULT = classes.DEFAULT_FILE.read_text()
PARSED = classes.parse(DEFAULT)
CLASSES, SETTINGS = PARSED["classes"], PARSED["settings"]
UNITS = {u["key"]: u for u in classes.run_units(CLASSES)}
W, H, F = 1000, 600, "f.jpg"


def det(box, score=0.9):
    x0, y0, x1, y1 = box
    return {"prompt": "p", "score": score, "box": list(box),
            "mask": np.ones((y1 - y0, x1 - x0), bool)}


@pytest.fixture
def proj(tmp_path):
    conn = db.connect(tmp_path / "state.sqlite", create=True)
    classes.save(tmp_path, DEFAULT)
    # urchins in reading order: 3 (top), 1, 2 (left to right on one row); a fish
    results.write_frame(tmp_path, F, W, H, SETTINGS, {
        "urchin": {"unit": UNITS["urchin"], "seconds": 0.1, "dets": [
            det((100, 300, 140, 340)), det((500, 300, 540, 340)), det((300, 50, 340, 90))]},
        "fish": {"unit": UNITS["fish"], "seconds": 0.1, "dets": [det((700, 400, 760, 430))]}})
    yield tmp_path, conn
    conn.close()


def test_black_urchin_is_a_third_type_in_the_urchin_group():
    black = next(c for c in CLASSES if c["name"] == "black urchin")
    assert black["id"] == 10 and black["sam_group"] == "urchin" and black["kind"] == "object"
    assert [(t["name"], t["key"]) for t in tagging.urchin_types(CLASSES)] == [
        ("purple urchin", "P"), ("red urchin", "R"), ("black urchin", "B")]
    # adding a type to the group does not re-run urchins: the settings key ignores names
    two = classes.parse(DEFAULT.replace(
        "  - {id: 10, name: black urchin, kind: object, prompts: [urchin], sam_group: urchin}\n",
        ""))["classes"]
    urchin2 = next(u for u in classes.run_units(two) if u["key"] == "urchin")
    assert results.settings_key(urchin2, SETTINGS) == results.settings_key(UNITS["urchin"],
                                                                           SETTINGS)


def test_type_keys_follow_the_name_and_clashes_get_none():
    renamed = [{**c, "name": {10: "rust urchin", 2: "rare urchin"}.get(c["id"], c["name"])}
               for c in CLASSES]
    assert [t["key"] for t in tagging.urchin_types(renamed)] == ["P", None, None]
    fuzzy = [{**c, "name": "zebra urchin"} if c["id"] == 10 else c for c in CLASSES]
    assert tagging.urchin_types(fuzzy)[2]["key"] is None           # Z is undo


def test_next_untagged_in_reading_order_wraps_and_skips_tagged(proj):
    folder, conn = proj
    fin = state.final(folder, conn, F, CLASSES)
    assert tagging.next_untagged(fin) == 3
    assert tagging.next_untagged(fin, 3) == 1 and tagging.next_untagged(fin, 1) == 2
    assert tagging.next_untagged(fin, 2) == 3                         # wraps
    assert tagging.next_untagged(fin, 4) == 3                         # after the fish: wraps
    tagging.tag(conn, folder, F, [1], 1, CLASSES)
    fin = state.final(folder, conn, F, CLASSES)
    assert tagging.next_untagged(fin, 3) == 2
    tagging.tag_all_remaining(conn, folder, F, 1, CLASSES)
    assert tagging.next_untagged(state.final(folder, conn, F, CLASSES)) is None


def test_tag_only_urchins_and_retag_between_types(proj):
    folder, conn = proj
    with pytest.raises(state.ReviewError, match="not an urchin"):
        tagging.tag(conn, folder, F, [4], 2, CLASSES)                 # the fish
    with pytest.raises(state.ReviewError, match="not an urchin type"):
        tagging.tag(conn, folder, F, [1], 6, CLASSES)
    tagging.tag(conn, folder, F, [1], 10, CLASSES)                    # B
    tagging.tag(conn, folder, F, [1], 2, CLASSES)                     # then R
    assert state.mask_rows(conn, F)[1]["class_id"] == 2


def test_tag_all_remaining_as_purple_is_one_undo_step(proj):
    folder, conn = proj
    tagging.tag(conn, folder, F, [2], 2, CLASSES)                     # the red outlier
    assert tagging.tag_all_remaining(conn, folder, F, 1, CLASSES) == 2
    rows = state.mask_rows(conn, F)
    assert {m: rows[m]["class_id"] for m in (1, 2, 3)} == {1: 1, 2: 2, 3: 1}
    state.accept(conn, folder, F, CLASSES)
    state.undo(conn, F)
    state.undo(conn, F)                                               # the "tag all" step
    assert state.final(folder, conn, F, CLASSES)["untagged"] == [1, 3]
    tagging.tag_all_remaining(conn, folder, F, 1, CLASSES)
    with pytest.raises(state.ReviewError, match="No untagged"):
        tagging.tag_all_remaining(conn, folder, F, 1, CLASSES)


def test_focus_shows_the_urchin_with_scene_inside_the_frame():
    assert tagging.focus([480, 280, 520, 320], W, H) == [300, 180, 700, 420]   # min 400 wide
    x0, y0, x1, y1 = tagging.focus([0, 0, 300, 100], W, H)
    assert (x0, y0) == (0, 0) and x1 - x0 == 900 and abs((y1 - y0) - 540) <= 1
    assert tagging.focus([0, 0, 900, 500], W, H) == [0, 0, W, H]


def test_a_rename_keeps_ids_results_and_review_without_a_re_run(proj):
    folder, conn = proj
    state.set_class(conn, folder, F, [4], 6, CLASSES)                 # fish 4 reviewed
    renamed = DEFAULT.replace("name: fish,", "name: rockfish,").replace(
        "name: black urchin,", "name: dark urchin,")
    data = classes.save(folder, renamed)
    raw = results.read(folder, F)
    assert "rockfish" in raw["units"] and "fish" not in raw["units"]
    assert raw["units"]["rockfish"]["classes"] == {"6": "rockfish"}
    assert [m["mask_id"] for m in raw["units"]["rockfish"]["masks"]] == [4]
    units = classes.run_units(data["classes"])
    assert {u["key"] for u in results.to_run(folder, F, units, SETTINGS)} == (
        {u["key"] for u in units} - {"urchin", "rockfish"})           # nothing re-runs
    fin = state.final(folder, conn, F, data["classes"])
    assert [(o["mask_id"], o["class_id"]) for o in fin["objects"] if o["unit"] == "rockfish"] \
        == [(4, 6)]
    assert tagging.urchin_types(data["classes"])[2] == {"id": 10, "name": "dark urchin",
                                                        "group": "urchin", "key": "D"}
    used = json.loads((folder / "sam_raw" / classes.USED_FILE).read_text())
    assert used["6"] == "rockfish"
    assert results.follow_renames(folder, data["classes"]) == 0       # idempotent
    # renaming back works too
    classes.save(folder, DEFAULT)
    assert "fish" in results.read(folder, F)["units"]


def test_a_swap_of_names_in_one_save_keeps_each_class_its_results(proj):
    """QE PR 47: fish (6) and sea star (3) swap names in one save; each id keeps its masks."""
    folder, conn = proj
    results.write_frame(folder, F, W, H, SETTINGS, {
        "sea star": {"unit": UNITS["sea star"], "seconds": 0.1, "dets": [det((10, 10, 30, 30))]}})
    star_ids = [m["mask_id"] for m in results.read(folder, F)["units"]["sea star"]["masks"]]
    fish_ids = [m["mask_id"] for m in results.read(folder, F)["units"]["fish"]["masks"]]
    swapped = DEFAULT.replace("name: fish,", "name: TMP,").replace(
        "name: sea star,", "name: fish,").replace("name: TMP,", "name: sea star,")
    data = classes.save(folder, swapped)
    raw = results.read(folder, F)
    assert raw["units"]["fish"]["classes"] == {"3": "fish"}
    assert [m["mask_id"] for m in raw["units"]["fish"]["masks"]] == star_ids
    assert raw["units"]["sea star"]["classes"] == {"6": "sea star"}
    assert [m["mask_id"] for m in raw["units"]["sea star"]["masks"]] == fish_ids
    units = classes.run_units(data["classes"])
    assert not {"fish", "sea star"} & {u["key"] for u in results.to_run(folder, F, units, SETTINGS)}



def test_two_saved_units_never_merge_into_one(proj):
    """Two saved units whose ids land in one current unit (red urchin pulled into the urchin
    group after it had its own results): the first keeps the key, the other is kept apart."""
    folder, conn = proj
    p = results.raw_path(folder, F)
    raw = json.loads(p.read_text())
    raw["units"]["red urchin"] = {**raw["units"]["fish"], "classes": {"2": "red urchin"}}
    p.write_text(json.dumps(raw))
    results.follow_renames(folder, CLASSES)
    after = results.read(folder, F)["units"]
    assert after["urchin"]["classes"] == {"1": "purple urchin", "2": "red urchin", "10": "black urchin"}
    assert after["red urchin"]["classes"] == {"2": "red urchin"}     # not merged, not lost
