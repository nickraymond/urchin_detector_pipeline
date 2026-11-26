"""S2 bite 5: folders of still images (copied, skip clips and extraction, join at review), and
every stage resuming after a stop or a restart. Synthetic images and video only."""

import sqlite3

import pytest
from PIL import Image

from app import db
from app.stages import extract, ingest, review
from tests import grab_hooks
from tests.test_app_skeleton import client_for, write_config
from tests.test_clips import wait_for
from tests.test_ingest import digest, make_video


def make_stills(root):
    """3 good stills (one PNG, one in a subfolder), a macOS ._ file, a broken JPEG and a
    hidden folder."""
    root.mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 30, 30)).save(root / "n6 frame 1.jpg", quality=95)
    Image.new("RGB", (64, 48), (30, 200, 30)).save(root / "pi_2.PNG")
    (root / "sub").mkdir()
    Image.new("RGB", (48, 64), (30, 30, 200)).save(root / "sub" / "pi_3.jpeg")
    (root / "._n6 frame 1.jpg").write_bytes(b"mac resource fork")
    (root / "broken.jpg").write_bytes(b"not a jpeg")
    (root / ".hidden").mkdir()
    Image.new("RGB", (8, 8)).save(root / ".hidden" / "x.jpg")
    return root


def _proj(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    return proj, db.connect(proj / "state.sqlite", create=True)


def test_add_stills_copies_unchanged_and_skips_repeats(tmp_path):
    src = make_stills(tmp_path / "stills")
    before = digest(src)
    proj, conn = _proj(tmp_path)
    assert ingest.add_stills(conn, src, extract.frames_dir(proj)) == {
        "added": 3, "skipped": 0, "unreadable": 1}
    rows = {r["video"]: dict(r) for r in conn.execute("SELECT * FROM frames")}
    assert set(rows) == {str(src / "n6 frame 1.jpg"), str(src / "pi_2.PNG"),
                         str(src / "sub" / "pi_3.jpeg")}
    for path, r in rows.items():
        assert r["status"] == "kept" and r["timestamp_ms"] is None
        assert extract.FRAME_NAME.match(r["file"]), r["file"]
        copy = extract.frames_dir(proj) / r["file"]
        assert copy.read_bytes() == open(path, "rb").read()          # byte for byte
    assert rows[str(src / "pi_2.PNG")]["file"].endswith("_0.png")
    assert rows[str(src / "n6 frame 1.jpg")]["file"].startswith("n6_frame_1_")
    assert digest(src) == before                                       # sources only read
    # again, and a same-name same-size copy in another folder: nothing new (D13)
    other = tmp_path / "copy"
    other.mkdir()
    (other / "pi_2.PNG").write_bytes((src / "pi_2.PNG").read_bytes())
    assert ingest.add_stills(conn, src, extract.frames_dir(proj))["added"] == 0
    assert ingest.add_stills(conn, other, extract.frames_dir(proj))["skipped"] == 1
    st = dict(conn.execute("SELECT stage, counts FROM stage_status"))
    assert st["ingest"] == "3 still images"


def test_a_truncated_jpeg_is_counted_unreadable_and_leaves_no_copy(tmp_path):
    src = tmp_path / "stills"
    src.mkdir()
    noise = Image.effect_noise((256, 256), 64).convert("RGB")
    noise.save(src / "full.jpg", quality=95)
    data = (src / "full.jpg").read_bytes()
    (src / "cut.jpg").write_bytes(data[: len(data) // 2])        # passes Image.verify()
    with Image.open(src / "cut.jpg") as im:
        im.verify()
    proj, conn = _proj(tmp_path)
    assert ingest.add_stills(conn, src, extract.frames_dir(proj)) == {
        "added": 1, "skipped": 0, "unreadable": 1}
    assert [p.name for p in extract.frames_dir(proj).iterdir()] == [
        ingest.still_name(src / "full.jpg")]                     # no orphan or temp copy
    assert ingest.add_stills(conn, src, extract.frames_dir(proj))["unreadable"] == 1


def test_too_many_stills_asks_for_a_subfolder(tmp_path, monkeypatch):
    src = make_stills(tmp_path / "stills")
    proj, conn = _proj(tmp_path)
    monkeypatch.setattr(ingest, "MAX_STILLS", 2)
    with pytest.raises(ingest.IngestError, match="subfolders"):
        ingest.add_stills(conn, src, extract.frames_dir(proj))


def test_stills_skip_extraction_and_filters_and_count_apart(tmp_path):
    src = make_stills(tmp_path / "stills")
    video = make_video(tmp_path / "v" / "dive.MP4", seconds=6, size="160x120")
    proj, conn = _proj(tmp_path)
    ingest.add_stills(conn, src, extract.frames_dir(proj))
    row = {"path": str(video), "path_code": "abc123", "duration_s": 6.0}
    plan, live = extract.sample_plan([row], [], 2.0, 300)
    extract.run(proj, plan, blur_percent=50, duplicate_distance=64)
    stills = [r[0] for r in conn.execute("SELECT file FROM frames WHERE timestamp_ms IS NULL")]
    assert len(stills) == 3
    assert all(s == "kept" for s, in conn.execute(
        "SELECT status FROM frames WHERE timestamp_ms IS NULL"))      # never filtered
    assert all((extract.frames_dir(proj) / f).is_file() for f in stills)  # never set aside
    c = extract.counts(conn)
    assert c["stills"] == 3 and c["total"] == live == 3
    assert c["kept"] + c["dropped_blur"] + c["dropped_duplicate"] == 3
    # review sees them; D moves the copy (never the source) to rejected/
    names = {f["file"] for f in review.frames(conn)}
    assert set(stills) <= names
    assert next(f for f in review.frames(conn) if f["file"] == stills[0])["t"] is None
    review.act(conn, proj, stills[0], "D")
    assert (proj / "rejected" / stills[0]).is_file()
    assert extract.counts(conn)["stills_rejected"] == 1
    extract.run(proj, plan)                                  # extract again: still rejected
    assert (proj / "rejected" / stills[0]).is_file()


def test_stills_through_the_ingest_and_review_pages(tmp_path):
    src = make_stills(tmp_path / "lib" / "stills")
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(tmp_path / "lib")]))
    client.post("/projects", data={"name": "p"})
    page = client.get("/p/p/ingest", params={"folder": str(src)}).text
    assert "Still images in this folder and its subfolders (4)" in page   # broken one listed
    r = client.post("/p/p/ingest/stills", data={"folder": str(src)})
    assert "Added 3 still images" in r.text and "1 could not be read and was left out" in r.text
    assert "3 still images in this project" in r.text
    assert client.post("/p/p/ingest/stills", data={"folder": str(tmp_path / "nope")}
                       ).status_code == 200                               # a message, no crash
    conn = sqlite3.connect(data / "projects" / "p" / "state.sqlite")
    png = conn.execute("SELECT file FROM frames WHERE file LIKE '%.png'").fetchone()[0]
    assert png in client.get("/p/p/review").text
    f = client.get(f"/p/p/frames/{png}")
    assert f.status_code == 200 and f.headers["content-type"] == "image/png"
    assert client.get(f"/p/p/framethumbs/{png}").status_code == 200
    assert "3 still images" in client.get("/p/p").text                   # dashboard ingest row


def test_restart_keeps_every_stage_and_extract_resumes(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    make_video(lib / "a.MP4", seconds=8, size="160x120")
    before = digest(lib)
    data = tmp_path / "data"
    data.mkdir()
    cfg = write_config(tmp_path, data, roots=[str(lib)])
    client = client_for(cfg)
    client.post("/projects", data={"name": "p"})
    client.post("/p/p/ingest/add", data={"folder": str(lib), "paths": [str(lib / "a.MP4")]})
    client.post("/p/p/extract/filter", data={"blur": "0", "dup": "0"})

    # a stop in the middle of extraction: the third frame fails (inside the worker process)
    monkeypatch.setattr(extract, "GRAB_HOOK", "tests.grab_hooks:fail_third")
    client.post("/p/p/extract/start", data={"every": "2", "cap": "300"})
    assert wait_for(lambda: client.get("/p/p/extract/status").json()["job"].get("state") == "error")
    folder = data / "projects" / "p"
    assert len(list((folder / "frames").glob("*.jpg"))) == 2

    # restart: a new app on the same settings file; nothing in memory survives
    monkeypatch.setattr(extract, "GRAB_HOOK", "tests.grab_hooks:counted")
    client.app.state.close_jobs()                              # the old app has quit
    client = client_for(cfg)
    dash = client.get("/p/p").text
    assert "Ingest" in dash and "1 video" in dash
    page = client.get("/p/p/extract").text
    assert "stopped" in page.lower() and "carry on" in page
    client.post("/p/p/extract/start", data={"every": "2", "cap": "300"})
    assert wait_for(lambda: client.get("/p/p/extract/status").json()["job"].get("state") == "done")
    assert grab_hooks.calls(folder) == 3 + 2                   # the first two not decoded again
    conn = sqlite3.connect(folder / "state.sqlite")
    files = [r[0] for r in conn.execute("SELECT file FROM frames ORDER BY timestamp_ms")]
    assert len(files) == 4

    client.post("/p/p/review/act", json={"file": files[1], "key": "D"})
    client.app.state.close_jobs()
    client = client_for(cfg)                                   # restart again
    states = dict(conn.execute("SELECT stage, state FROM stage_status"))
    assert (states["ingest"], states["extract"], states["review"]) == \
        ("done", "done", "in progress")
    assert (folder / "rejected" / files[1]).is_file()
    assert client.post("/p/p/review/act", json={"key": "Z"}).json()["status"] == "kept"
    # re-running Extract redoes nothing and leaves review decisions alone
    client.post("/p/p/review/act", json={"file": files[1], "key": "D"})
    monkeypatch.setattr(extract, "GRAB_HOOK", "tests.grab_hooks:never")
    client.post("/p/p/extract/start", data={"every": "2", "cap": "300"})
    assert wait_for(lambda: client.get("/p/p/extract/status").json()["job"].get("state") == "done")
    assert (folder / "rejected" / files[1]).is_file()
    assert digest(lib) == before                               # SPEC constraint 4
