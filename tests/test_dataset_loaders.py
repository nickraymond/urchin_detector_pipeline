"""S8 bite 2b: what training will see (app/datasets/view.py) and the detector loader
(app/datasets/loaders.py): "fully checked" and the split check are honored. Synthetic exports."""

import pytest

from app.datasets import loaders, versions, view
from tests.dataset_fixtures import dive, fake_export


def frozen(tmp_path):
    """One dive in train with four kinds of frame; smaller dives in valid and test."""
    reef = dive("reef", 1, 14)                                  # urchin + star checked
    reef[0].update(split_ok="no")                               # cover only
    reef[1].update(checked=[1], stars=2)                        # stars labeled, not checked
    reef[2].update(checked=[])                                  # nothing checked
    reef[3].update(checked=[1, 3], urchins=0)                   # a real "no urchin" frame
    fake_export(tmp_path, reef + dive("cove", 2, 4) + dive("wall", 3, 2, checked=[3]))
    versions.freeze(tmp_path)
    return reef


def test_loader_skips_unchecked_classes_and_split_no_frames(tmp_path):
    reef = frozen(tmp_path)
    got = loaders.detector_samples(tmp_path, 1, "train", [1, 3])
    by = {s["file"]: s for s in got["samples"]}
    why = {x["file"]: x["why"] for x in got["left_out"]}
    assert "split check: no" in why[reef[0]["file"]] and reef[0]["file"] not in by
    assert "fully checked" in why[reef[2]["file"]] and reef[2]["file"] not in by
    s1 = by[reef[1]["file"]]            # urchins are targets; the unchecked stars are ignored
    assert [b[0] for b in s1["boxes"]] == [1, 1] and s1["checked"] == [1]
    assert len(s1["ignore"]) == 2 and all(len(b) == 4 for b in s1["ignore"])
    s3 = by[reef[3]["file"]]            # checked with none: kept, as a true negative
    assert s3["boxes"] == [] and s3["checked"] == [1, 3] and s3["ignore"] == []
    assert len(got["samples"]) == 12 and s1["boxes"][0][1:] == [4, 4, 5, 5]
    loaders.verify_image(s1)
    # the class choice is per run: a run on sea stars only does not see or ignore urchins
    stars = loaders.detector_samples(tmp_path, 1, "train", [3])
    assert reef[1]["file"] not in {s["file"] for s in stars["samples"]}
    assert all(b[0] == 3 for s in stars["samples"] for b in s["boxes"])
    assert all(s["ignore"] == [] for s in stars["samples"])
    # the test split's frames have only sea star checked: an urchin-only run gets none of them
    assert loaders.detector_samples(tmp_path, 1, "test", [1])["samples"] == []
    assert len(loaders.detector_samples(tmp_path, 1, "test", [3])["samples"]) == 2


def test_default_classes_are_sea_stars_and_urchins_by_name(tmp_path):
    frozen(tmp_path)
    manifest, _ = loaders.load(tmp_path, 1)
    assert loaders.default_class_ids(manifest) == [1, 3]       # the fixture has these two
    assert loaders.detector_samples(tmp_path, 1, "train")["class_ids"] == [1, 3]
    with pytest.raises(loaders.DatasetError, match="not object classes"):
        loaders.detector_samples(tmp_path, 1, "train", [9])    # kelp is an area
    with pytest.raises(loaders.DatasetError, match="at least one"):
        loaders.detector_samples(tmp_path, 1, "train", [])
    with pytest.raises(loaders.DatasetError, match="cannot be read"):
        loaders.load(tmp_path, 7)


def test_a_changed_image_store_is_caught(tmp_path):
    frozen(tmp_path)
    s = loaders.detector_samples(tmp_path, 1, "valid")["samples"][0]
    with open(s["image"], "ab") as fh:
        fh.write(b"x")
    with pytest.raises(loaders.DatasetError, match="no longer matches its hash"):
        loaders.verify_image(s)


def test_view_counts_what_the_loader_gives_and_what_is_skipped(tmp_path):
    reef = frozen(tmp_path)
    v = view.training_view(tmp_path, 1)
    urchin, star = v["classes"]
    assert (urchin["name"], star["name"]) == ("purple urchin", "sea star")
    # D66 (loaders.run_frames): a frame with both classes fully checked trains in full; one with
    # only some of them checked trains on its labeled boxes only; none checked: left out.
    # train: 14 frames - split no - reef[2] (nothing checked) = 12; reef[1] (stars unchecked) is
    # boxes-only, so 11 are fully checked. The urchins of reef[1] still count (22 objects).
    assert urchin["seen"]["train"] == {"objects": 22, "frames": 11}
    assert urchin["seen"]["valid"] == {"objects": 8, "frames": 4}
    assert urchin["seen"]["test"] == {"objects": 0, "frames": 0}
    assert urchin["skipped"]["train"] == {"objects": 2, "frames": 1}     # reef[2]
    assert urchin["skipped"]["test"] == {"objects": 4, "frames": 2}      # wall: stars only
    assert star["seen"]["train"] == {"objects": 0, "frames": 11}
    assert star["skipped"]["train"] == {"objects": 2, "frames": 2}       # reef[1], reef[2]
    assert urchin["total"] == 30 and urchin["under_target"] and v["target"] == (150, 200)
    assert v["split_no"] == [reef[0]["file"]]
    assert v["frames"]["train"] == {"used": 11, "boxes_only": 1, "left_out": 2}
    assert v["frames"]["test"] == {"used": 0, "boxes_only": 2, "left_out": 0}
    listed = {u["file"]: u["classes"] for u in v["unchecked"]}
    assert listed[reef[1]["file"]] == ["sea star"]
    assert listed[reef[2]["file"]] == ["purple urchin", "sea star"]
    assert reef[0]["file"] not in listed and len(listed) == 4            # + wall's two
    # the view and the loader agree for any class choice
    assert v["unseen"] == ["sea star"]       # 2 stars labeled in train, on unchecked frames only
    one = view.training_view(tmp_path, 1, [3])
    assert [c["name"] for c in one["classes"]] == ["sea star"]
    # urchins alone: reef[1] is a fully checked frame (12), nothing is boxes-only
    alone = view.training_view(tmp_path, 1, [1])
    assert alone["classes"][0]["seen"]["train"] == {"objects": 22, "frames": 12}
    assert alone["frames"]["train"] == {"used": 12, "boxes_only": 0, "left_out": 2}
    # the page's frame counts are the trainer's, for any class choice (QE PR 72)
    from app.train import job as train_job
    assert train_job.usable is loaders.usable
    for ids, got in (([1, 3], v), ([1], alone), ([3], one)):
        for split in ("train", "valid", "test"):
            samples = loaders.detector_samples(tmp_path, 1, split, ids)["samples"]
            full, boxes_only = loaders.run_frames(samples, ids)
            assert got["frames"][split]["used"] == len(full) == len(loaders.usable(samples, ids)[0])
            assert got["frames"][split]["boxes_only"] == len(boxes_only)
            assert all(0 < len(s["checked"]) < len(ids) for s in boxes_only)
    assert [c["picked"] for c in one["all_classes"]] == [False, True]


def test_a_labeled_class_with_no_fully_checked_frame_is_called_out(tmp_path):
    """D64: nothing fully checked (the real S5 project's state): the view names the classes
    that would train on 0 objects, the page says what to tick, and Freeze still works."""
    from tests.test_preview import make_project
    client, folder = make_project(tmp_path, 1)
    fake_export(folder, dive("reef", 1, 6, checked=[], stars=1) + dive("cove", 2, 2, checked=[])
                + dive("wall", 3, 1, checked=[]))
    assert versions.freeze(folder)["new"]
    v = view.training_view(folder, 1)
    assert v["unseen"] == ["purple urchin", "sea star"]
    page = client.get("/p/p/datasets").text
    assert "Training would see 0 objects of purple urchin, sea star" in page
    assert 'tick "fully checked"' in page and 'href="/p/p/labels"' in page
