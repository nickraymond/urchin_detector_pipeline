"""S2 bite 2a: background jobs, playback copies, thumbnail strips, clips page.
Synthetic videos made by ffmpeg at test time; no real footage."""

import re
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
    src = make_video(tmp_path / "v" / "dive.MP4", seconds=12, size="1280x720")
    before = digest(tmp_path / "v")
    row = {"path": str(src), "path_code": "abc123", "duration_s": 12.0}
    seen = []
    out = clips.make_playback(row, tmp_path / "p", lambda p, m: seen.append(p))
    assert probe(out) == "h264,960,540"  # scaled down to 960 px wide
    assert [f.name for f in clips.strip_frames(tmp_path / "p", "abc123")] == [
        "0001.jpg", "0002.jpg", "0003.jpg"]  # one frame at 0, 5 and 10 s
    assert seen and max(seen) <= 100 and clips.ready(tmp_path / "p", "abc123")
    # macOS shadow files on the SSD (Nick's test) must not show up as strip images
    (clips.strip_dir(tmp_path / "p", "abc123") / "._0001.jpg").write_bytes(b"\0" * 4096)
    assert len(clips.strip_frames(tmp_path / "p", "abc123")) == 3
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

    page = client.get("/p/p/clips").text  # a GET starts nothing (QE finding 2 on PR 5)
    assert "Make playback copies (1 video)" in page
    time.sleep(0.5)
    assert client.get("/p/p/clips/status").json()[code] == {"ready": False}
    r = client.post("/p/p/clips/prepare", data={"v": code},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403  # only the app's own page can start the work (D14)
    r = client.post("/p/p/clips/prepare", data={"v": code}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/p/p/clips?v={code}"
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


def test_strip_times_do_not_drift():
    """QE finding 1 on PR 5: thumbnail n is at n * 5 s (within one frame), even after 10 min."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "t.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc=size=64x48:rate=30000/1001", "-t", "601", "-pix_fmt", "yuv420p",
                        str(src)], check=True)
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(src), "-vf",
                            clips.strip_filter() + ",showinfo", "-fps_mode", "vfr", "-f", "null",
                            "-"], capture_output=True, text=True)
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", r.stderr)]
    assert len(times) == 121
    assert all(0 <= t - n * clips.STRIP_EVERY_S < 1001 / 30000 + 1e-6 for n, t in enumerate(times))


def test_lrv_in_another_codec_is_reencoded(tmp_path):
    """QE finding 3 on PR 5: only an H.264 .LRV is repackaged as it is."""
    src = make_video(tmp_path / "v" / "GX010002.MP4", seconds=6, size="320x240")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
                    "-t", "6", "-c:v", "mpeg4", "-f", "mp4", str(tmp_path / "v" / "GL010002.LRV")],
                   check=True)
    assert clips.codec_of(tmp_path / "v" / "GL010002.LRV") == "mpeg4"
    out = clips.make_playback({"path": str(src), "path_code": "lrv002", "duration_s": 6.0},
                              tmp_path / "p")
    assert probe(out) == "h264,64,48"  # from the proxy, H.264, never scaled up
