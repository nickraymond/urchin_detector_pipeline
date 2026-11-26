"""S4 bite 6c (Nick's S4 test, items 9, 9a, 8a): the Labeling page's "Labeled so far" grid
(every kept frame with results, newest first, its counts; the data polled cheaply while Run
goes on) and the project page's SAM labeling row counted from the results on disk. Fake SAM,
synthetic frames."""

import json
import os

from app.sam import results
from app.stages import label
from tests.test_preview import make_project
from tests.test_run import run_and_wait
from tests.test_sam_runner import FAKE


def test_labeled_frames_are_the_kept_ones_with_results_newest_first(tmp_path):
    raw = results.raw_dir(tmp_path)
    raw.mkdir()
    for i, name in enumerate(["a.png", "b.png", "c.png", "gone.png"]):
        p = raw / f"{name}.json"
        p.write_text("{}")
        os.utime(p, ns=(10**18 + i * 10**9,) * 2)
    (raw / "class_ids.json").write_text("{}")
    assert label.labeled_frames(tmp_path, ["a.png", "b.png", "c.png", "d.png"]) == [
        ("c.png", 10**18 + 2 * 10**9), ("b.png", 10**18 + 10**9), ("a.png", 10**18)]
    assert label.labeled_frames(tmp_path / "none", ["a.png"]) == []


def test_labeled_so_far_on_the_page_and_in_its_poll(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path, n=4)
    page = client.get("/p/p/labeling").text
    assert 'id="labeled"' in page and "0 of 4 kept frames labeled" in page
    assert ">Labeled so far<" in page
    proj = client.get("/p/p").text
    assert "frames labeled" not in proj                      # nothing on disk yet
    run_and_wait(client)
    d = client.get("/p/p/labeling/labeled").json()
    assert (d["total"], d["kept"], d["shown"]) == (4, 4, 4)
    assert sorted(f["file"] for f in d["frames"]) == [f"v_aaaaaa_{i}.png" for i in range(4)]
    f = d["frames"][0]
    assert f["counts"] and all(isinstance(n, int) for n in f["counts"].values())  # per unit
    assert f["v"] and f["older"] == []
    assert client.get(f"/p/p/labeling/labeled?since={d['sig']}").json() == {
        "sig": d["sig"], "same": True}                        # polling costs a listing only
    two = client.get("/p/p/labeling/labeled?n=2").json()
    assert two["shown"] == 2 and two["total"] == 4 and two["sig"] != d["sig"]
    page = client.get("/p/p/labeling?n=2").text
    assert "4 of 4 kept frames labeled; the newest 2 are shown." in page
    assert '<a href="?n=50#labeled">Show 48 more</a>' in page
    data = json.loads(page.split('id="labeled-list">')[1].split("</script>")[0])
    assert data["poll"] == "/p/p/labeling/labeled?n=2" and data["data"]["shown"] == 2
    # 8a: the project page counts the results on disk
    row = client.get("/p/p").text.split("SAM labeling</a>")[1].split("</tr>")[0]
    assert "<td>done" in row and "4 of 4 frames labeled" in row
    # the viewer and its frame list are on the page without a preview too
    page = client.get("/p/p/labeling").text
    assert 'id="viewer"' in page and 'id="frame-list"' in page
