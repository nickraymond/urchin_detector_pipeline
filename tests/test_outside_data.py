"""S8 bite 3: outside data (app/datasets/outside.py). A tiny synthetic "outside" folder in the
earlier work's layout; no real dataset is read."""

import json

import pytest
from PIL import Image

from app.datasets import outside


def make_outside(root):
    """urchinbot: 4 images (2 train, 1 valid, 1 test; one also listed in valid and test);
    rf100: prefix splits; duo and an unknown folder to be refused."""
    ub = root / "urchinbot"
    (ub / "images").mkdir(parents=True)
    (ub / "archives").mkdir()
    rows = []
    for i, n in enumerate((2, 0, 1, 3)):
        Image.new("RGB", (40, 30), (i, 0, 0)).save(ub / "images" / f"im{i}.JPG")
        rows.append({"file": f"images/im{i}.JPG", "w": 40, "h": 30, "classes": ["urchin"],
                     "boxes": [[0, 1 + j, 2, 5, 6, 30] for j in range(n)]})
    (ub / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ub / "archives" / "train.txt").write_text("data/images/im0.JPG\ndata/images/im1.JPG\n")
    (ub / "archives" / "val.txt").write_text("data/images/im2.JPG\ndata/images/im3.JPG")
    (ub / "archives" / "test.txt").write_text("data/images/im3.JPG")   # in both: test wins
    rf = root / "roboflow" / "rf100_underwater"
    for s in ("train", "valid", "test"):
        (rf / s / "images").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(rf / s / "images" / "a.jpg")
    (rf / "labels.jsonl").write_text("".join(json.dumps(
        {"file": f"{s}/images/a.jpg", "w": 8, "h": 8, "classes": ["urchin"],
         "boxes": [[0, 1, 1, 2, 2, 4]]}) + "\n" for s in ("train", "valid", "test")))
    (root / "duo").mkdir()
    (root / "mystery_set").mkdir()
    return root


def test_read_uses_train_and_valid_only_and_one_class(tmp_path):
    root = make_outside(tmp_path)
    got = outside.read(root, "urchinbot")
    assert got["counts"] == {"train": {"images": 2, "boxes": 2}, "valid": {"images": 1, "boxes": 1}}
    assert got["unused"] == {"test": 1, "no_split": 0} and got["on_disk"] == {"images": 4, "boxes": 6}
    assert {r["split"] for r in got["rows"]} == {"train", "valid"}
    assert got["rows"][0]["boxes"] == [[1, 2, 5, 6], [2, 2, 5, 6]]       # x, y, w, h; no class
    rf = outside.read(root, "rf100_underwater")
    assert [r["split"] for r in rf["rows"]] == ["train", "valid"] and rf["unused"]["test"] == 1


def test_sources_without_a_commercial_verdict_are_refused(tmp_path):
    root = make_outside(tmp_path)
    with pytest.raises(outside.OutsideError, match="DUO set stays out"):
        outside.read(root, "duo")
    with pytest.raises(outside.OutsideError, match="not an allowed outside source"):
        outside.read(root, "mystery_set")
    for key in ("fathomnet", "noaa_hf_weights", "corpus_v1", "corpus_v2", "gbif_inat",
                "urchin_detector_74"):
        assert not outside.BY_KEY[key]["allowed"] and outside.BY_KEY[key]["why_not"]
    cat = {c["key"]: c for c in outside.catalogue(root)}
    assert cat["urchinbot"]["can_add"] and cat["rf100_underwater"]["can_add"]
    assert cat["duo"]["present"] and not cat["duo"]["can_add"]
    assert cat["mystery_set"]["licence"] == "no record" and not cat["mystery_set"]["can_add"]
    assert not cat["fathomnet"]["present"]
    assert all(c["licence"] for c in cat.values())
    assert outside.catalogue(None)[0]["can_add"] is False      # no folder set


def test_a_box_of_another_class_stops_the_read(tmp_path):
    root = make_outside(tmp_path)
    p = root / "urchinbot" / "labels.jsonl"
    p.write_text(p.read_text() + json.dumps(
        {"file": "images/im0.JPG", "w": 4, "h": 4, "classes": ["starfish"],
         "boxes": [[0, 1, 1, 2, 2, 4]]}) + "\n")
    with pytest.raises(outside.OutsideError, match="starfish"):
        outside.read(root, "urchinbot")


def test_snapshot_and_samples(tmp_path):
    root = make_outside(tmp_path / "outside")
    vdir = tmp_path / "v1"
    (vdir / "outside").mkdir(parents=True)
    entry, data = outside.snapshot(root, "urchinbot")
    (vdir / "outside" / entry["rows"]).write_bytes(data)
    assert entry["licence"] == "CC-BY-4.0" and "zenodo" in entry["attribution"].lower()
    assert entry["counts"]["train"] == {"images": 2, "boxes": 2} and len(entry["sha256"]) == 64
    assert entry["class"] == "urchin" and "never the test split" in entry["use"]
    train = outside.samples(root, vdir, entry, "train")
    assert [s["file"] for s in train] == ["images/im0.JPG", "images/im1.JPG"]
    assert train[0]["bytes"] == (root / "urchinbot" / "images" / "im0.JPG").stat().st_size
    assert train[1]["boxes"] == [] and (train[0]["width"], train[0]["height"]) == (40, 30)
    assert len(outside.samples(root, vdir, entry, "valid")) == 1
    assert outside.samples(root, vdir, entry, "test") == []     # never the test split
    (root / "urchinbot" / "images" / "im1.JPG").unlink()
    with pytest.raises(outside.OutsideError, match="missing"):
        outside.snapshot(root, "urchinbot")


# ---- a dataset version with an outside source; the page; starting weights ----

def test_a_version_can_add_an_allowed_source_and_never_a_refused_one(tmp_path):
    from app.datasets import versions
    from tests.dataset_fixtures import dive, fake_export
    root = make_outside(tmp_path / "outside")
    proj = tmp_path / "proj"
    fake_export(proj, dive("reef", 1, 6) + dive("cove", 2, 2) + dive("wall", 3, 1))
    plain = versions.freeze(proj)["manifest"]
    assert plain["outside"] == [] and not (proj / "datasets" / "v1" / "outside").exists()
    for key, msg in (("duo", "DUO set stays out"), ("mystery_set", "not an allowed")):
        with pytest.raises(versions.FreezeError, match=msg):
            versions.freeze(proj, outside_keys=[key], outside_dir=root)
    with pytest.raises(versions.FreezeError, match="No outside data folder"):
        versions.freeze(proj, outside_keys=["urchinbot"])
    assert [v["number"] for v in versions.list_versions(proj)] == [1]
    res = versions.freeze(proj, outside_keys=["urchinbot"], outside_dir=root)
    m = res["manifest"]
    assert res["new"] and res["number"] == 2 and m["hash"] != plain["hash"]   # content changed
    (entry,) = m["outside"]
    rows = proj / "datasets" / "v2" / "outside" / "urchinbot.jsonl"
    assert versions.sha256_file(rows) == entry["sha256"] and entry["licence"] == "CC-BY-4.0"
    assert {f["file"]: f["split"] for f in m["frames"]} == \
           {f["file"]: f["split"] for f in plain["frames"]}        # Nick's frames untouched
    assert outside.samples(root, proj / "datasets" / "v2", entry, "test") == []
    assert len(outside.samples(root, proj / "datasets" / "v2", entry, "train")) == 2
    assert not list((proj / "datasets" / "images").glob("im*"))    # outside images not copied
    assert not versions.freeze(proj, outside_keys=["urchinbot"], outside_dir=root)["new"]


def test_page_lists_sources_with_licences_and_adds_one(tmp_path):
    from tests.dataset_fixtures import dive, fake_export
    from tests.test_app_skeleton import client_for, write_config
    from tests.test_clips import wait_for
    data = tmp_path / "data"
    data.mkdir()
    root = make_outside(tmp_path / "outside")
    cfg = write_config(tmp_path, data)
    cfg.write_text(cfg.read_text() + f'\noutside_data_dir = "{root}"\n')
    client = client_for(cfg)
    client.post("/projects", data={"name": "p"})
    fake_export(data / "projects" / "p", dive("reef", 1, 3))
    page = client.get("/p/p/datasets").text
    assert 'name="outside_keys" value="urchinbot"' in page and "CC-BY-4.0" in page
    assert 'value="duo"' not in page and "the DUO set stays out" in page
    assert "No licence record for this folder" in page and "same image lineage as the DUO" in page
    r = client.post("/p/p/datasets/freeze", data={"outside_keys": "duo"}, follow_redirects=False)
    assert "cannot+be+added" in r.headers["location"]
    client.post("/p/p/datasets/freeze", data={"outside_keys": "urchinbot"})
    assert wait_for(lambda: "Version 1" in client.get("/p/p/datasets").text, 60)
    page = client.get("/p/p/datasets").text
    assert "Outside data in this version" in page and "<td>2 (2)</td><td>1 (1)</td>" in page
    assert "1 of its own test images" in page and "doi.org/10.5281/zenodo.16060266" in page
    assert "on disk: 4 images / 6 boxes" in page                 # differs from the record: shown


def test_page_without_the_setting_says_how_to_set_it(tmp_path):
    from tests.test_preview import make_project
    client, folder = make_project(tmp_path, 1)
    page = client.get("/p/p/datasets").text
    assert "No folder is set" in page and 'name="outside_keys"' not in page


def test_starting_weights_are_a_checked_recorded_choice():
    from app.train import starting
    assert starting.record(None) == {"kind": "scratch", "licences": []}
    with pytest.raises(starting.StartError, match="No published weights are offered"):
        starting.record({"kind": "published", "name": "yolox_nano.pth"})
    models = {"m1": {"outside": [{"key": "urchinbot"}], "start": {"kind": "scratch"}},
              "m2": {"outside": [{"key": "duo"}]}, "m3": {"outside": []},
              "m4": {"outside": [{"key": "rf100_underwater"}],
                     "start": {"kind": "pretrained", "licences": ["CC-BY-4.0"]}},
              "m5": {"outside": [{"key": "urchinbot"}], "start": {"kind": "published"}}}
    got = starting.record({"kind": "pretrained", "model": "m1"}, models)
    assert got == {"kind": "pretrained", "model": "m1", "licences": ["CC-BY-4.0"],
                   "sources": [{"key": "urchinbot", "licence": "CC-BY-4.0"}]}
    assert starting.record({"kind": "pretrained", "model": "m4"}, models)["licences"] == \
           ["CC BY 4.0", "CC-BY-4.0"]                      # the chain's licences are kept
    for mid, msg in (("m2", "not an allowed outside source"), ("m3", "not trained on outside"),
                     ("m5", "not a clean"), ("nope", "No model")):
        with pytest.raises(starting.StartError, match=msg):
            starting.record({"kind": "pretrained", "model": mid}, models)
    with pytest.raises(starting.StartError, match="Unknown starting weights"):
        starting.record({"kind": "magic"})
    assert not starting.PUBLISHED and len(starting.NOT_OFFERED) == 2
