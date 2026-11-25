"""S3 bite 3: prompt preview (D26): frame choice, the background job through the page, the
overlay with per-class toggles, counts and timing. Fake SAM, synthetic frames only."""

import sqlite3

from PIL import Image

from app.stages import label
from tests.test_app_skeleton import client_for, write_config
from tests.test_clips import wait_for
from tests.test_sam_runner import FAKE, synthetic_frame


def test_preview_frames_first_middle_last_or_random():
    files = [f"f{i}.jpg" for i in range(7)]
    assert label.preview_frames(files) == ["f0.jpg", "f3.jpg", "f6.jpg"]
    assert label.preview_frames(files[:2]) == ["f0.jpg", "f1.jpg"]
    import random
    other = label.preview_frames(files, other=True, rng=random.Random(1))
    assert len(other) == 3 and set(other) <= set(files) and other == sorted(other)


def make_project(tmp_path, n=4):
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data))
    client.post("/projects", data={"name": "p"})
    folder = data / "projects" / "p"
    (folder / "frames").mkdir()
    conn = sqlite3.connect(folder / "state.sqlite")
    for i in range(n):
        name = f"v_aaaaaa_{i}.png"
        synthetic_frame(folder / "frames" / name)
        conn.execute("INSERT INTO frames (file, video, timestamp_ms, status) VALUES (?, 'v', ?, "
                     "'kept')", (name, i))
    conn.commit()
    return client, folder


def test_preview_through_the_page(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path)
    r = client.post("/p/p/labeling/preview", data={})
    assert r.status_code == 200
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "done", 60)
    page = client.get("/p/p/labeling").text
    assert "s per frame" in page and "Run on all 4 kept frames" in page
    for f in ("v_aaaaaa_0.png", "v_aaaaaa_2.png", "v_aaaaaa_3.png"):
        assert f in page
    assert "urchin: 2" in page and "sea star: 1" in page and "fish: 0" in page
    full = client.get("/p/p/labeling/overlay/v_aaaaaa_0.png")
    assert full.status_code == 200 and full.headers["content-type"] == "image/jpeg"
    hidden = client.get("/p/p/labeling/overlay/v_aaaaaa_0.png", params={"hide": "urchin,kelp"})
    assert hidden.content != full.content
    assert client.get("/p/p/labeling/overlay/v_aaaaaa_1.png").status_code == 404  # not run
    assert client.get("/p/p/labeling/overlay/..%2Fstate.sqlite").status_code == 404
    # a changed class shows its old counts as older settings until it runs again
    text = (folder / "classes.yaml").read_text().replace("prompts: [fish]", "prompts: [rock]")
    client.post("/p/p/labeling/classes", data={"text": text})
    assert "fish: 0 (older settings)" in client.get("/p/p/labeling").text


def test_preview_refused_without_frames_or_with_bad_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path, n=0)
    assert "No kept frames to preview" in client.post("/p/p/labeling/preview").text
    (folder / "classes.yaml").write_text("classes: []\n")
    assert "classes.yaml has problems" in client.post("/p/p/labeling/preview").text


def test_overlay_draws_masks_on_the_upright_frame(tmp_path):
    import io

    from app.sam import classes
    client, folder = make_project(tmp_path, n=1)
    label.run(folder, ["v_aaaaaa_0.png"], lambda p, m: None, predictor=FAKE)
    units = classes.run_units(classes.ensure(folder)["classes"])

    def draw(hidden):
        return Image.open(io.BytesIO(label.overlay(folder, "v_aaaaaa_0.png", units, hidden)))
    shown, bare = draw(set()), draw({"urchin"})
    assert shown.size == (1600, 738)
    at = (round(230 * 1600 / 2600),) * 2              # the urchin at (230, 230) in the frame
    assert shown.getpixel(at) != bare.getpixel(at)     # drawn in untagged grey over magenta
    assert shown.getpixel((50, 50)) == bare.getpixel((50, 50))   # open water untouched
    assert shown.getpixel(at)[1] > bare.getpixel(at)[1]          # grey lifts the green
