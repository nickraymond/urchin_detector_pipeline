"""S3 bite 1: classes.yaml (checks with line numbers, run units, the Labeling page) and the
EXIF-orientation helper (D27). Synthetic images only."""

import json
import sqlite3

import pytest
import yaml
from PIL import Image

from app import images
from app.sam import classes
from app.stages import review
from tests.test_app_skeleton import client_for, write_config

DEFAULT = classes.DEFAULT_FILE.read_text()


def errors_of(text, used=None):
    with pytest.raises(classes.ClassesError) as e:
        classes.parse(text, used)
    return e.value.errors


def test_default_file_is_valid_with_the_planned_ids_and_order():
    data = classes.parse(DEFAULT)
    by_id = {c["id"]: c["name"] for c in data["classes"]}
    assert by_id == {1: "purple urchin", 2: "red urchin", 3: "sea star",
                     4: "sunflower sea star", 5: "anemone", 6: "fish", 7: "rock/substrate",
                     8: "kelp shoot", 9: "kelp"}
    names = [c["name"] for c in data["classes"]]
    assert names.index("kelp") < names.index("rock/substrate")          # D22: kelp wins
    assert data["settings"] == {"model": "sam3", "confidence": 0.4, "tile_px": 1008,
                                "tile_overlap": 0.2, "tile_above_px": 2000,
                                "min_mask_area_px": None}
    shoot = next(c for c in data["classes"] if c["name"] == "kelp shoot")
    assert shoot["seg_as"] == "kelp" and shoot["kind"] == "object"


def test_urchins_are_one_run_unit_with_one_prompt():
    units = classes.run_units(classes.parse(DEFAULT)["classes"])
    urchin = units[0]
    assert urchin == {"key": "urchin", "group": True, "kind": "object", "prompts": ["urchin"],
                      "classes": {1: "purple urchin", 2: "red urchin"}}
    assert len(units) == 8 and all(not u["group"] for u in units[1:])


def test_errors_name_the_line():
    text = DEFAULT.replace("{id: 3, name: sea star, kind: object",
                           "{id: 2, name: sea star, kind: thing")
    errs = errors_of(text)
    line = next(i for i, s in enumerate(text.splitlines(), 1) if "name: sea star" in s)
    assert (line, "id 2 is used twice (also line %d)" % (line - 1)) in errs
    assert (line, "kind must be object or area; got 'thing'") in errs


def test_yaml_syntax_error_names_the_line():
    errs = errors_of("settings:\n  model: sam3\nclasses: [\n  {id: 1\n")
    assert errs[0][0] is not None and "not valid YAML" in errs[0][1]


@pytest.mark.parametrize("old, new, expect", [
    ("prompts: [urchin], sam_group: urchin}\n  - {id: 2, name: red urchin, kind: object, "
     "prompts: [urchin]", "prompts: [urchin], sam_group: urchin}\n  - {id: 2, name: red urchin,"
     " kind: object, prompts: [red urchin]", "need the same prompts"),
    ("{id: 2, name: red urchin, kind: object", "{id: 2, name: red urchin, kind: area",
     "must be objects"),
    ("seg_as: kelp}", "seg_as: rock}", "seg_as must name an area class"),
    ("{id: 6, name: fish,", "{id: 0, name: fish,", "0 means unknown"),
    ("{id: 6, name: fish,", "{id: 6, name: Kelp,", "is used twice"),
    ("prompts: [fish]}", "prompts: []}", "prompts must be a list"),
    ("prompts: [fish]}", "prompts: [fish], colour: red}", "unknown key 'colour'"),
    ("confidence: 0.4", "confidence: 1.5", "confidence must be"),
    ("tile_px: 1008", "tile_pix: 1008", "unknown setting 'tile_pix'"),
    ("{id: 6, name: fish, kind: object, prompts: [fish]}",
     "{id: 6, name: urchin, kind: object, prompts: [fish]}", "has the name of another class"),
])
def test_bad_classes_are_refused(old, new, expect):
    assert old in DEFAULT
    errs = errors_of(DEFAULT.replace(old, new))
    assert any(expect in m and n for n, m in errs), errs


def test_ids_with_sam_results_keep_their_names():
    used = {3: "sea star", 6: "fish"}
    errs = errors_of(DEFAULT.replace("name: sea star,", "name: bat star,"), used)
    assert any("id 3 has SAM results as 'sea star'" in m and "e.g. 10" in m for _, m in errs)
    errs = errors_of(DEFAULT.replace("{id: 6, name: fish", "{id: 10, name: fish"), used)
    assert any("'fish' has SAM results under id 6" in m for _, m in errs)
    # removing a class with results is fine; its results stay on disk (DESIGN)
    classes.parse(DEFAULT.replace("  - {id: 6, name: fish, kind: object, prompts: [fish]}\n", ""),
                  used)


def test_save_is_safe_dump_and_nothing_is_written_on_error(tmp_path):
    with pytest.raises(classes.ClassesError):
        classes.save(tmp_path, "classes: []")
    assert not (tmp_path / "classes.yaml").exists()
    data = classes.save(tmp_path, DEFAULT)
    written = (tmp_path / "classes.yaml").read_text()
    assert yaml.safe_load(written) == data and "#" not in written
    (tmp_path / "sam_raw").mkdir()
    (tmp_path / "sam_raw" / "class_ids.json").write_text(json.dumps({"3": "sea star"}))
    with pytest.raises(classes.ClassesError):
        classes.save(tmp_path, DEFAULT.replace("name: sea star,", "name: bat star,"))
    assert (tmp_path / "classes.yaml").read_text() == written


def test_ensure_copies_the_starting_file_once(tmp_path):
    classes.ensure(tmp_path)
    assert (tmp_path / "classes.yaml").read_text() == DEFAULT
    (tmp_path / "classes.yaml").write_text(DEFAULT.replace("[fish]", "[fishes]"))
    assert "fishes" in str(classes.ensure(tmp_path))


def test_labeling_page_shows_edits_saves_and_refuses(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data))
    client.post("/projects", data={"name": "p"})
    folder = data / "projects" / "p"
    assert 'href="/p/p/labeling"' in client.get("/p/p").text           # dashboard row
    page = client.get("/p/p/labeling").text
    assert "not in the project yet" in page and "sunflower sea star" in page
    assert "Next free id: 10" in page
    assert not (folder / "classes.yaml").exists()                        # a GET writes nothing
    bad = DEFAULT.replace("kind: area, prompts: [kelp]", "kind: area, prompts: kelp")
    r = client.post("/p/p/labeling/classes", data={"text": bad})
    assert r.status_code == 400 and "Not saved" in r.text and "Line 22:" in r.text
    assert "prompts: kelp" in r.text                                     # what was typed stays
    assert not (folder / "classes.yaml").exists()
    r = client.post("/p/p/labeling/classes",
                    data={"text": DEFAULT + "  - {id: 10, name: sea cucumber, kind: object, "
                                            "prompts: [sea cucumber]}\n"})
    assert "Saved classes.yaml." in r.text and "sea cucumber" in r.text
    assert "Next free id: 11" in r.text
    # edited in a text editor, then Reload: the page shows the problem with its line
    (folder / "classes.yaml").write_text(DEFAULT.replace("confidence: 0.4", "confidence: 2"))
    page = client.get("/p/p/labeling").text
    assert "cannot be used yet" in page and "Line 5: confidence must be" in page


# --- EXIF orientation (D27) ---

def rotated_jpeg(path, size=(80, 40)):
    """A landscape JPEG whose EXIF says 'rotate 90° clockwise to view' (orientation 6)."""
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", size, (10, 120, 200)).save(path, exif=exif)
    return path


def test_open_upright_applies_exif_orientation(tmp_path):
    p = rotated_jpeg(tmp_path / "r.jpg")
    assert Image.open(p).size == (80, 40)
    assert images.open_upright(p).size == (40, 80)
    Image.new("RGB", (80, 40)).save(tmp_path / "plain.jpg")
    assert images.open_upright(tmp_path / "plain.jpg").size == (80, 40)


def test_review_thumbnail_of_a_rotated_still_is_upright(tmp_path):
    proj = tmp_path / "p"
    (proj / "frames").mkdir(parents=True)
    rotated_jpeg(proj / "frames" / "s_aaaaaa_0.jpg", size=(1280, 640))
    out = review.thumbnail(proj, "s_aaaaaa_0.jpg")
    assert Image.open(out).size == (640, 1280)                           # portrait, 640 px wide
    assert out.parent == proj / "framethumbs" / "upright"
