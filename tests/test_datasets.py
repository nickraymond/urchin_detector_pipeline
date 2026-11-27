"""S8 bite 2a: dataset versions (app/datasets): freeze from an export, manifest, hash, and
every group keeping its split across versions (D60, D61). Synthetic exports only."""

import json

import pytest

from app.datasets import splits as ds_splits
from app.datasets import versions
from tests.dataset_fixtures import dive, fake_export, frame


def tree(p):
    return {str(f.relative_to(p)): f.read_bytes() for f in sorted(p.rglob("*")) if f.is_file()}


def four_dives():
    return dive("reef", 1, 14) + dive("cove", 2, 4) + dive("wall", 3, 2) + dive("arch", 4, 1)


def test_freeze_writes_a_version_with_manifest_counts_and_hash(tmp_path):
    frames = four_dives()
    frames[0]["split_ok"] = "no"                      # cover only: not detector data
    export = fake_export(tmp_path, frames)
    res = versions.freeze(tmp_path, "  first   run ")
    m = res["manifest"]
    assert res["new"] and res["number"] == 1 and m["name"] == "first run" and m["previous"] is None
    v1 = tmp_path / "datasets" / "v1"
    assert json.loads((v1 / "manifest.json").read_text()) == m
    assert not list((tmp_path / "datasets").glob("*.new"))
    # frames by reference: each image once, under its content hash, bytes equal to the export's
    for f in m["frames"]:
        stored = tmp_path / "datasets" / "images" / f["image"]
        assert stored.read_bytes() == (export / "images" / f["file"]).read_bytes()
        assert versions.sha256_file(stored) == f["image"].split(".")[0]
    # version 1 takes the export's splits as they are; whole groups, all three splits used
    want = {s: (export / "splits" / f"{s}.txt").read_text() for s in ("train", "valid", "test")}
    assert {s: (v1 / "splits" / f"{s}.txt").read_text() for s in want} == want
    assert {g["carried"] for g in m["groups"]} == {"export"} and len(m["groups"]) == 4
    assert m["counts"]["frames"] == {"train": 15, "valid": 4, "test": 2}
    assert m["counts"]["detector_frames"]["train"] == 14
    urchin = next(c for c in m["counts"]["classes"] if c["name"] == "purple urchin")
    assert urchin["objects"] == {"train": 28, "valid": 8, "test": 4}
    assert urchin["frames"] == {"train": 14, "valid": 4, "test": 2}
    # the label files are recorded by hash and the hash is the content's
    for rel, sha in m["files"].items():
        assert versions.sha256_file(v1 / rel) == sha
    assert m["hash"] == versions.content_hash(m) and len(m["hash"]) == 64


def test_same_export_again_writes_no_new_version_and_changes_nothing(tmp_path):
    fake_export(tmp_path, four_dives())
    versions.freeze(tmp_path)
    before = tree(tmp_path / "datasets")
    fake_export(tmp_path, four_dives(), at="2026-11-28T09:00:00")    # a re-export, same content
    res = versions.freeze(tmp_path, "another name")
    assert not res["new"] and res["number"] == 1
    assert tree(tmp_path / "datasets") == before


def test_groups_keep_their_split_when_the_export_deals_them_again(tmp_path):
    """The export deals its groups again once the data changes (D43); the version does not."""
    fake_export(tmp_path, four_dives())
    v1 = versions.freeze(tmp_path)["manifest"]
    old = {f["file"]: f["split"] for f in v1["frames"]}
    assert {g["group"]: g["split"] for g in v1["groups"]}["reef 2025-10-01"] == "train"
    # a much bigger new dive: the export alone would now put "cove" somewhere else
    frames = four_dives() + dive("pinnacle", 5, 40) + dive("reef", 1, 16)[14:]
    export = fake_export(tmp_path, frames)
    moved = [f for f in old if f not in (export / "splits" / f"{old[f]}.txt").read_text()]
    assert moved, "the fixture must make the export re-deal at least one old frame"
    before = tree(tmp_path / "datasets" / "v1")
    res = versions.freeze(tmp_path)
    v2 = res["manifest"]
    assert res["number"] == 2 and v2["previous"] == 1
    now = {f["file"]: f["split"] for f in v2["frames"]}
    assert all(now[f] == s for f, s in old.items())               # nothing moved
    groups = {g["group"]: g for g in v2["groups"]}
    assert groups["reef 2025-10-01"]["carried"] == "v1" and groups["reef 2025-10-01"]["frames"] == 16
    assert all(now[f["file"]] == "train" for f in frames if "reef" in f["file"])   # joined
    assert groups["pinnacle 2025-10-05"]["carried"] == "new"
    coco = json.loads((tmp_path / "datasets" / "v2" / "detector_coco.json").read_text())
    assert all(i["split"] == now[i["file_name"]] for i in coco["images"])
    assert tree(tmp_path / "datasets" / "v1") == before            # versions are never changed
    assert len(list((tmp_path / "datasets" / "images").iterdir())) == len(frames)   # stored once


def test_new_groups_are_dealt_toward_70_20_10_from_the_carried_counts():
    prev = {"number": 3, "frames": [{"file": f"a{i}", "split": "train"} for i in range(10)],
            "groups": [{"group": "A", "split": "train", "frames": 10}]}
    frames = [{"file": f"a{i}", "group": "A", "split": "test"} for i in range(10)]
    frames += [{"file": f"b{i}", "group": "B", "split": "train"} for i in range(4)]
    frames += [{"file": f"c{i}", "group": "C", "split": "train"} for i in range(2)]
    by_frame, groups = ds_splits.assign(frames, prev)
    got = {g["group"]: (g["split"], g["carried"]) for g in groups}
    assert got == {"A": ("train", "v3"), "B": ("valid", "new"), "C": ("test", "new")}
    assert by_frame["a0"] == "train" and by_frame["b0"] == "valid" and by_frame["c1"] == "test"


def test_a_dive_that_straddled_splits_keeps_its_test_frames_out_of_training():
    """D61: v1 grouped by video (fewer than 3 dives) and put one dive's two videos in train
    and test; later the export groups by site + date. Old frames stay; new ones go to test."""
    prev = {"number": 1,
            "frames": [{"file": "x1", "split": "train"}, {"file": "x2", "split": "test"}],
            "groups": [{"group": "/v/a.MP4", "split": "train", "frames": 1},
                       {"group": "/v/b.MP4", "split": "test", "frames": 1}]}
    frames = [{"file": f, "group": "reef 2025-10-01", "split": "train"} for f in ("x1", "x2", "x3")]
    by_frame, groups = ds_splits.assign(frames, prev)
    assert by_frame == {"x1": "train", "x2": "test", "x3": "test"}
    assert groups == [{"group": "reef 2025-10-01", "frames": 3, "split": "test", "carried": "v1",
                       "straddles": ["test", "train"]}]


def test_freeze_refuses_without_a_good_export_and_writes_nothing(tmp_path):
    with pytest.raises(versions.FreezeError, match="No export yet"):
        versions.freeze(tmp_path)
    export = fake_export(tmp_path, four_dives())
    s = json.loads((export / "summary.json").read_text())
    (export / "summary.json").write_text(json.dumps({**s, "ok": False}))
    with pytest.raises(versions.FreezeError, match="did not pass its checks"):
        versions.freeze(tmp_path)
    assert not (tmp_path / "datasets").exists()


def test_a_stopped_freeze_leaves_no_version_and_the_next_one_cleans_up(tmp_path):
    fake_export(tmp_path, four_dives())

    def stop(i, n, name):
        if i == 3:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        versions.freeze(tmp_path, progress=stop)
    assert versions.list_versions(tmp_path) == [] and (tmp_path / "datasets" / "v1.new").is_dir()
    assert versions.freeze(tmp_path)["number"] == 1
    assert [p.name for p in (tmp_path / "datasets").iterdir() if p.name != "images"] == ["v1"]


def test_stills_and_the_video_fallback_group_like_the_export():
    assert ds_splits.group_of("", "still_abc123_0.png", "site and dive date") == "still_abc123_0.png"
    src = "/videos/reef-25-10-01-0900-xx-GOPR.MP4"
    assert ds_splits.group_of(src, "f.png", "site and dive date") == "reef 2025-10-01"
    assert ds_splits.group_of(src, "f.png", "video") == src
    assert frame(src, 0)["file"].endswith("_0.png")
