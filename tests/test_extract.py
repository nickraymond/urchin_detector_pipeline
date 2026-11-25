"""S2 bite 3a: sampling plan, cap, extraction, resume, Extract page.
Synthetic videos made by ffmpeg at test time; no real footage."""

import sqlite3

import pytest
from PIL import Image

from app import db
from app.stages import extract, ingest
from tests.test_app_skeleton import client_for, write_config
from tests.test_clips import wait_for
from tests.test_ingest import digest, make_video

A = {"path": "/v/a.MP4", "path_code": "aaaaaa", "duration_s": 10.0, "width": 64, "height": 48}
B = {"path": "/v/South Wreck b.MP4", "path_code": "bbbbbb", "duration_s": 7.0, "width": 64,
     "height": 48}


def test_plan_clips_whole_video_and_live_count():
    clips = [{"video": "/v/a.MP4", "start_s": 6, "end_s": 9, "note": ""},
             {"video": "/v/a.MP4", "start_s": 1, "end_s": 2.5, "note": ""}]
    samples, live = extract.sample_plan([A, B], clips, 2.0, 300)
    assert [(s["video"]["path_code"], s["t"]) for s in samples] == [
        ("aaaaaa", 1.0), ("aaaaaa", 6.0), ("aaaaaa", 8.0),          # inside a's clips, in order
        ("bbbbbb", 0.0), ("bbbbbb", 2.0), ("bbbbbb", 4.0), ("bbbbbb", 6.0)]  # b has none: whole
    assert live == 7
    # the live count matches bite 2b's clip page count for the same clips
    from app.stages import clips as clip_stage
    assert clip_stage.summary([A, B], clips, 2.0)["frames"] == live


def test_cap_spreads_evenly_across_videos():
    samples, live = extract.sample_plan([A, B], [], 0.5, 10)
    assert live == 34 and len(samples) == 10
    per = [s["video"]["path_code"] for s in samples]
    assert per.count("aaaaaa") == 6 and per.count("bbbbbb") == 4  # share kept (20 vs 14 samples)
    assert samples[0]["t"] == 0.0 and samples[-1]["t"] == 6.5     # first and last sample kept


def test_frame_name():
    assert extract.frame_name("/v/South Wreck-25 f1.MP4", "abc123", 12.3456) == \
        "South_Wreck-25_f1_abc123_12346.jpg"
    assert extract.FRAME_NAME.match("South_Wreck-25_f1_abc123_12346.jpg")


def test_run_extracts_full_size_and_resumes(tmp_path):
    src = make_video(tmp_path / "v" / "dive.MP4", seconds=5, size="320x240")
    before = digest(tmp_path / "v")
    row = {"path": str(src), "path_code": "abc123", "duration_s": 5.0, "width": 320,
           "height": 240}
    proj = tmp_path / "p"
    proj.mkdir()
    db.connect(proj / "state.sqlite", create=True).close()
    samples, _ = extract.sample_plan([row], [], 2.0, 300)
    assert extract.run(proj, samples) == 3
    files = sorted(p.name for p in extract.frames_dir(proj).iterdir())
    assert files == ["dive_abc123_0.jpg", "dive_abc123_2000.jpg", "dive_abc123_4000.jpg"]
    with Image.open(extract.frames_dir(proj) / files[1]) as im:
        assert im.size == (320, 240) and im.format == "JPEG"  # full size from the original
    conn = sqlite3.connect(proj / "state.sqlite")
    rows = conn.execute("SELECT file, video, timestamp_ms, blur_score, hash, status "
                        "FROM frames ORDER BY timestamp_ms").fetchall()
    assert [r[2] for r in rows] == [0, 2000, 4000] and {r[1] for r in rows} == {str(src)}
    assert all(r[3] > 0 and len(r[4]) == 16 and r[5] == "kept" for r in rows)
    assert conn.execute("SELECT state, counts FROM stage_status WHERE stage='extract'"
                        ).fetchone() == ("done", "3 frames")
    (extract.frames_dir(proj) / files[0]).unlink()  # a frame lost: the next run makes it again
    assert extract.run(proj, samples) == 1
    assert not list(extract.frames_dir(proj).glob("*.tmp"))
    assert digest(tmp_path / "v") == before  # SPEC constraint 4


def test_blur_score_is_lower_for_a_blurred_frame():
    sharp = Image.effect_mandelbrot((640, 360), (-2, -1, 1, 1), 50).convert("RGB")
    from PIL import ImageFilter
    assert extract.blur_score(sharp.filter(ImageFilter.GaussianBlur(4))) < extract.blur_score(sharp)
    assert extract.dhash(sharp) == extract.dhash(sharp.resize((320, 180)))  # same picture


def test_missing_video_is_a_clear_error(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    db.connect(proj / "state.sqlite", create=True).close()
    gone = {**A, "path": str(tmp_path / "gone.MP4")}
    with pytest.raises(FileNotFoundError, match="Plug in the SSD"):
        extract.run(proj, [{"video": gone, "t": 0.0}])


def test_extract_page_start_progress_and_frames(tmp_path):
    lib = tmp_path / "videos"
    make_video(lib / "a.MP4", seconds=4, size="320x240")
    before = digest(lib)
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(lib)]))
    client.post("/projects", data={"name": "p"})
    assert "No videos in this project yet" in client.get("/p/p/extract").text
    client.post("/p/p/ingest/add", data={"folder": str(lib), "paths": [str(lib / "a.MP4")]})
    page = client.get("/p/p/extract").text
    assert "2 frames from 1 video" in page  # 4 s at one every 2 s
    assert client.get("/p/p/extract/plan?every=1&cap=3").json() == {"live": 4, "planned": 3}
    assert client.get("/p/p/extract/plan?every=0&cap=3").status_code == 400

    r = client.post("/p/p/extract/start", data={"every": "1", "cap": "3"},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    client.post("/p/p/extract/filter", data={"blur": "0", "dup": "0"})  # filters off (bite 3b)
    r = client.post("/p/p/extract/start", data={"every": "1", "cap": "3"}, follow_redirects=False)
    assert r.status_code == 303
    assert wait_for(lambda: client.get("/p/p/extract/status").json()["job"].get("state") == "done")
    assert client.get("/p/p/extract/status").json()["counts"]["kept"] == 3
    page = client.get("/p/p/extract").text
    assert "<b>3</b> kept" in page and 'value="1.0"' in page  # settings saved to project.json
    assert 'id="viewer"' in page and page.count('data-i="') == 3  # in-page viewer (Nick)
    import json, re
    listed = json.loads(re.search(r'id="frame-list">(.*?)</script>', page).group(1))
    assert [f["t"] for f in listed["frames"]] == [0.0, 2.0, 3.0]  # cap 3 of 0,1,2,3 s
    code = ingest.path_code(lib / "a.MP4")
    r = client.get(f"/p/p/frames/a_{code}_0.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    for bad in ("..%2Fproject.json", f"a_{code}_999.jpg", "x.jpg"):
        assert client.get(f"/p/p/frames/{bad}").status_code == 404, bad
    assert "3 frames" in client.get("/p/p").text
    assert digest(lib) == before


def _project(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    db.connect(proj / "state.sqlite", create=True).close()
    return proj


def test_new_settings_set_earlier_frames_aside_and_bring_them_back(tmp_path):
    """QE PR 7 finding 1: kept always equals the current plan; nothing is erased."""
    src = make_video(tmp_path / "v" / "dive.MP4", seconds=10, size="160x120")
    row = {"path": str(src), "path_code": "abc123", "duration_s": 10.0}
    proj = _project(tmp_path)
    conn = sqlite3.connect(proj / "state.sqlite")
    kept = lambda: sorted(r[0] for r in conn.execute("SELECT file FROM frames WHERE status='kept'"))

    every5, _ = extract.sample_plan([row], [], 5.0, 300)   # 0, 5
    every2, _ = extract.sample_plan([row], [], 2.0, 300)   # 0, 2, 4, 6, 8
    cap3, _ = extract.sample_plan([row], [], 2.0, 3)       # 0, 4, 8
    assert extract.run(proj, every5) == 2
    assert extract.run(proj, every2) == 4                  # 0 reused, 5 set aside
    assert kept() == sorted(extract.frame_name(src, "abc123", s["t"]) for s in every2)
    assert (extract.unplanned_dir(proj) / "dive_abc123_5000.jpg").is_file()
    assert extract.run(proj, cap3) == 0
    c = extract.counts(conn)
    assert c["kept"] == 3 and c["unplanned"] == 3 and c["total"] == 3
    assert sorted(p.name for p in extract.frames_dir(proj).glob("*.jpg")) == kept()
    assert extract.run(proj, every5) == 0                  # 5 s comes back, not re-extracted
    assert kept() == ["dive_abc123_0.jpg", "dive_abc123_5000.jpg"]
    assert len(list(extract.unplanned_dir(proj).glob("*.jpg"))) == 4


def test_rotated_video_comes_out_upright(tmp_path):
    """QE PR 7 finding 2: a video with rotation metadata (phone, upright GoPro) decodes with
    width and height swapped; the frame must be saved, not fail."""
    flat = make_video(tmp_path / "flat.MP4", seconds=2, size="320x240")
    rot = tmp_path / "v" / "upright.MP4"
    rot.parent.mkdir()
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(flat),
                    "-c", "copy", "-y", str(rot)], check=True)
    row = {"path": str(rot), "path_code": "abc123", "duration_s": 2.0, "width": 320,
           "height": 240}
    proj = _project(tmp_path)
    assert extract.run(proj, [{"video": row, "t": 1.0}]) == 1
    with Image.open(extract.frames_dir(proj) / "upright_abc123_1000.jpg") as im:
        assert im.size == (240, 320)


def test_a_stop_between_a_move_and_its_database_update_is_repaired(tmp_path):
    """QE PR 7 at 730b58e: a frame moved but not yet updated in the database comes right on
    the next run, in both directions, without decoding it again."""
    src = make_video(tmp_path / "v" / "dive.MP4", seconds=4, size="160x120")
    row = {"path": str(src), "path_code": "abc123", "duration_s": 4.0}
    proj = _project(tmp_path)
    plan, _ = extract.sample_plan([row], [], 2.0, 300)          # 0, 2
    assert extract.run(proj, plan) == 2
    name = "dive_abc123_2000.jpg"
    conn = sqlite3.connect(proj / "state.sqlite")
    status = lambda: conn.execute("SELECT status FROM frames WHERE file=?", (name,)).fetchone()[0]
    aside = extract.unplanned_dir(proj)
    aside.mkdir(parents=True)
    # stopped after moving back to frames/, before the row said 'kept'
    conn.execute("UPDATE frames SET status='unplanned' WHERE file=?", (name,)); conn.commit()
    assert extract.run(proj, plan) == 0 and status() == "kept"
    # stopped after moving aside, before the row said 'unplanned'; the plan wants it again
    (extract.frames_dir(proj) / name).rename(aside / name)
    assert extract.run(proj, plan) == 0 and status() == "kept"
    assert (extract.frames_dir(proj) / name).is_file() and not (aside / name).exists()
