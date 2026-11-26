"""S5 bite 5a: add a missed object (D46): crop region, request checks, the add worker with a fake
click / box predictor, saving under the unit `added`, and how review and re-runs treat it."""

import json

import pytest

from app import db
from app.review import state
from app.sam import add, add_worker, classes, results
from app.stages import label
from tests.test_sam_runner import synthetic_frame

CLASSES = classes.parse(classes.DEFAULT_FILE.read_text())["classes"]
FAKE = "tests.fake_sam:FakeClickPredictor"
F = "v_aaaaaa_0.png"   # 2600 x 1200: urchins at (960-1060, 300-400) and (200-260, 200-260),
                       # a sea star at (2000-2100, 500-600), a kelp band y 1000-1200


def test_region_is_a_tile_around_the_spot_inside_the_frame():
    assert add.region_for(2600, 1200, point=[1010, 350]) == (506, 0, 1514, 1008)
    assert add.region_for(2600, 1200, point=[5, 5]) == (0, 0, 1008, 1008)
    assert add.region_for(2600, 1200, point=[2599, 1199]) == (1592, 192, 2600, 1200)
    assert add.region_for(800, 600, point=[10, 10]) == (0, 0, 800, 600)       # small frame: whole
    assert add.region_for(2600, 1200, box=[0, 0, 1000, 700]) == (0, 0, 2000, 1200)


@pytest.mark.parametrize("kw, msg", [({}, "not both"), ({"point": [1, 1], "box": [0, 0, 9, 9]}, "not both"),
                                     ({"point": [2600, 5]}, "outside"),
                                     ({"box": [10, 10, 12, 40]}, "at least 4")])
def test_bad_requests_are_refused(kw, msg):
    with pytest.raises(ValueError, match=msg):
        add.check_request(2600, 1200, **kw)


def project(tmp_path):
    (tmp_path / "frames").mkdir()
    synthetic_frame(tmp_path / "frames" / F)
    conn = db.connect(tmp_path / "state.sqlite", create=True)
    conn.execute("INSERT INTO frames (file, video, timestamp_ms, status) VALUES (?, 'v', 0, 'kept')",
                 (F,))
    conn.commit()
    return conn


@pytest.fixture
def capsys(monkeypatch):
    """The worker's event lines (app/jobctl/worker.py binds stdout when imported)."""
    import io

    from app.jobctl import worker as jw
    buf = io.StringIO()
    monkeypatch.setattr(jw, "OUT", buf)

    class Cap:
        def readouterr(self):
            out = buf.getvalue()
            buf.seek(0)
            buf.truncate()
            return type("R", (), {"out": out})
    return Cap()


def run_add(tmp_path, capsys, **kw):
    params = label.add_params(tmp_path, F, kw.pop("class_id", 1), **kw)
    params["predictor"] = FAKE
    job = label.build_add_job(params)
    add_worker.main(job)
    events = [json.loads(ln[3:]) for ln in capsys.readouterr().out.splitlines() if ln.startswith("@@ ")]
    return params, events


def test_a_click_adds_the_object_under_it_as_reviewed_and_by_hand(tmp_path, capsys):
    conn = project(tmp_path)
    params, ev = run_add(tmp_path, capsys, point=[230, 230])      # the small urchin
    assert ev[-1]["event"] == "done" and "added mask 1" in ev[-1]["message"]
    raw = results.read(tmp_path, F)
    (m,) = raw["units"]["added"]["masks"]
    assert m["box"] == [200, 200, 261, 261] and m["class_id"] == 1
    assert m["reviewed"] and m["added_by_hand"] and m["request"] == params["request"]
    assert label.add_progress(params) == (1, 1)
    fin = state.final(tmp_path, conn, F, CLASSES)
    assert [(o["mask_id"], o["class_id"], o["added"]) for o in fin["objects"]] == [(1, 1, True)]
    # the same request again (a resumed job) adds nothing
    add_worker.main(label.build_add_job({**params, "predictor": FAKE}))
    assert "already added" in capsys.readouterr().out
    assert len(results.read(tmp_path, F)["units"]["added"]["masks"]) == 1


def test_a_box_adds_the_largest_object_inside_and_water_adds_nothing(tmp_path, capsys):
    project(tmp_path)
    _, ev = run_add(tmp_path, capsys, box=[1980, 480, 2120, 620], class_id=3)   # the sea star
    assert "added mask" in ev[-1]["message"]
    assert results.read(tmp_path, F)["units"]["added"]["masks"][0]["box"] == [2000, 500, 2101, 601]
    _, ev = run_add(tmp_path, capsys, point=[1500, 100])                       # open water
    assert ev[-1]["message"].startswith("SAM found nothing there")
    assert len(results.read(tmp_path, F)["units"]["added"]["masks"]) == 1


def test_an_added_mask_survives_a_re_run_and_wins_over_a_fresh_duplicate(tmp_path, capsys):
    conn = project(tmp_path)
    run_add(tmp_path, capsys, point=[1010, 350], class_id=2)                  # red urchin by hand
    units = {u["key"]: u for u in classes.run_units(CLASSES)}
    s = classes.parse(classes.DEFAULT_FILE.read_text())["settings"]
    import numpy as np
    det = {"prompt": "urchin", "score": 0.99, "box": [960, 300, 1061, 401],
           "mask": np.ones((101, 101), bool)}
    results.write_frame(tmp_path, F, 2600, 1200, s,
                        {"urchin": {"unit": units["urchin"], "seconds": 0.1, "dets": [det]}},
                        state.reviewed_ids(tmp_path, F))
    raw = results.read(tmp_path, F)
    assert len(raw["units"]["added"]["masks"]) == 1                          # untouched by the run
    fin = state.final(tmp_path, conn, F, CLASSES)
    assert [(o["class_id"], o["added"]) for o in fin["objects"]] == [(2, True)]   # D31: reviewed wins
    assert fin["untagged"] == []


def test_the_add_job_uses_sam3_and_the_kind_cannot_pause(tmp_path):
    project(tmp_path)
    job = label.build_add_job(label.add_params(tmp_path, F, 1, point=[1, 1]))
    assert job["checkpoint"] == label.MODELS["sam3"][1] and job["path"].endswith(F)
    assert label.add_kind().pausable is False and label.add_kind().name == "sam_add"
    with pytest.raises(label.LabelError, match="missing"):
        label.build_add_job(label.add_params(tmp_path, "nope.png", 1, point=[1, 1]))
