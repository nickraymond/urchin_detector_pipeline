"""S2 bite 2a: background jobs, playback copies, thumbnail strips, clips page.
Synthetic videos made by ffmpeg at test time; no real footage."""

import shutil
import subprocess
import threading
import time

import pytest

from app.jobs import Jobs
from app.stages import clips, ingest
from tests.test_app_skeleton import client_for, write_config
from tests.test_ingest import digest, make_video


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return r.stdout.strip()


def wait_for(check, seconds=60):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.2)
    return False


# --- jobs ---

def test_jobs_run_report_and_survive_errors():
    jobs = Jobs()
    gate = threading.Event()

    def slow(report):
        report(50, "half")
        gate.wait(5)

    jobs.submit("a", slow)
    assert wait_for(lambda: jobs.status("a")["percent"] == 50)
    jobs.submit("a", slow)  # already running: not queued twice
    jobs.submit("b", lambda report: 1 / 0)
    jobs.submit("c", lambda report: None)
    gate.set()
    assert wait_for(lambda: jobs.status("c")["state"] == "done")
    assert jobs.status("a")["state"] == "done"
    assert jobs.status("b")["state"] == "error" and "ZeroDivisionError" in jobs.status("b")["message"]
    assert jobs.status("zzz") is None


# --- playback copy and strip ---

def test_copy_and_strip(tmp_path):
    src = make_video(tmp_path / "v" / "dive.MP4", seconds=12, size="320x240")
    before = digest(tmp_path / "v")
    row = {"path": str(src), "path_code": "abc123", "duration_s": 12.0}
    seen = []
    out = clips.make_playback(row, tmp_path / "p", lambda p, m: seen.append(p))
    assert probe(out) == "h264,960,720"  # scaled to 960 px wide
    assert [f.name for f in clips.strip_frames(tmp_path / "p", "abc123")] == [
        "0001.jpg", "0002.jpg", "0003.jpg"]  # one frame at 0, 5 and 10 s
    assert seen and max(seen) <= 100 and clips.ready(tmp_path / "p", "abc123")
    assert not list((tmp_path / "p").rglob("*.tmp*"))  # no temporary files left
    assert digest(tmp_path / "v") == before  # SPEC constraint 4


def test_gopro_lrv_is_repackaged_not_reencoded(tmp_path):
    src = make_video(tmp_path / "v" / "GX010001.MP4", seconds=6, size="320x240")
    make_video(tmp_path / "lrv" / "proxy.mp4", seconds=6, size="64x48")
    shutil.move(tmp_path / "lrv" / "proxy.mp4", tmp_path / "v" / "GL010001.LRV")
    out = clips.make_playback({"path": str(src), "path_code": "lrv001", "duration_s": 6.0},
                              tmp_path / "p")
    assert probe(out) == "h264,64,48"


def test_missing_source_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Plug in the SSD"):
        clips.make_playback({"path": str(tmp_path / "gone.MP4"), "path_code": "000000",
                             "duration_s": 1.0}, tmp_path / "p")


# --- pages ---

def test_clips_page_copy_player_and_files(tmp_path):
    lib = tmp_path / "videos"
    make_video(lib / "a.MP4", seconds=6, size="320x240")
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(lib)]))
    client.post("/projects", data={"name": "p"})
    assert "No videos in this project yet" in client.get("/p/p/clips").text
    client.post("/p/p/ingest/add", data={"folder": str(lib), "paths": [str(lib / "a.MP4")]})
    code = ingest.path_code(lib / "a.MP4")

    page = client.get("/p/p/clips").text  # queues the copy
    assert "Making the playback copy" in page or "<video" in page
    assert wait_for(lambda: client.get("/p/p/clips/status").json()[code]["ready"])
    page = client.get("/p/p/clips").text
    assert f'src="/p/p/playback/{code}.mp4"' in page and "/strip/" in page

    r = client.get(f"/p/p/playback/{code}.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100  # seeking works
    assert client.get(f"/p/p/strip/{code}/0001.jpg").content[:2] == b"\xff\xd8"
    for bad in (f"/p/p/playback/ffffff.mp4", f"/p/p/strip/{code}/..%2Fx.jpg",
                f"/p/p/strip/{code}/1.jpg", f"/p/p/strip/ffffff/0001.jpg"):
        assert client.get(bad).status_code == 404, bad
    assert "Pick clips</a>" in client.get("/p/p").text
