"""S2 bite 1b: ingest. Synthetic videos made by ffmpeg at test time; no real footage."""

import hashlib
import shutil
import subprocess
from urllib.parse import urlencode

import pytest

from app import db
from app.stages import ingest
from tests.test_app_skeleton import client_for, write_config


def make_video(path, seconds=1, size="64x48"):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size={size}:rate=10", "-t", str(seconds), "-pix_fmt", "yuv420p",
                    "-y", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A folder that looks like an SSD: GoPro chapters, a ReefCheck-style name, a copy in a
    second folder, a broken file, sidecars, a macOS shadow file and a hidden folder."""
    root = tmp_path_factory.mktemp("videos")
    base = make_video(root / "dive" / "GX010042.MP4", seconds=2)
    shutil.copy(base, root / "dive" / "GX020042.MP4")
    (root / "dive" / "GL010042.LRV").write_bytes(b"proxy")
    (root / "dive" / "GX010042.THM").write_bytes(b"thumb")
    (root / "dive" / "._GX010042.MP4").write_bytes(b"\0" * 4096)
    reef = make_video(root / "site" / "Site-25-08-29-1230-Diver-c2.MP4", seconds=3)
    (root / "copy").mkdir()
    shutil.copy(reef, root / "copy" / reef.name)
    (root / "site" / "broken.mov").write_bytes(b"not a video" * 100)
    make_video(root / ".hidden" / "x.mp4")
    return root


def digest(root):
    """sha256 and modified time of every file, to prove nothing was changed."""
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_gopro_names():
    from pathlib import Path
    assert ingest.recording_of(Path("GX020042.MP4")) == ("GX0042", 2)
    assert ingest.recording_of(Path("gh010007.mp4")) == ("GH0007", 1)
    assert ingest.recording_of(Path("Site-25-08-29-1230-Diver-c2.MP4")) == (
        "Site-25-08-29-1230-Diver-c2", 1)


def test_scan_rules(library):
    videos = {v.path.name: v for v in ingest.scan(library)}
    # sidecars, ._ files and hidden folders skipped; the copy counted once
    assert set(videos) == {"GX010042.MP4", "GX020042.MP4", "Site-25-08-29-1230-Diver-c2.MP4",
                           "broken.mov"}
    reef = videos["Site-25-08-29-1230-Diver-c2.MP4"]
    assert reef.also_in and reef.codec == "h264" and (reef.width, reef.height) == (64, 48)
    assert reef.duration_s == pytest.approx(3, abs=0.2) and reef.fps == 10
    assert len(reef.code) == 6 and not reef.warning
    one, two = videos["GX010042.MP4"], videos["GX020042.MP4"]
    assert one.recording == two.recording and (one.chapter, two.chapter) == (1, 2)
    assert one.has_lrv and not reef.has_lrv
    assert "cannot read" in videos["broken.mov"].warning


def test_add_remove_and_status(library, tmp_path):
    conn = db.connect(tmp_path / "s.sqlite", create=True)
    n = ingest.add(conn, ingest.scan(library))
    assert n == 3  # the broken video is never added
    assert db.stage_statuses(conn)[0]["counts"] == "3 videos (2 recordings)"
    ingest.remove(conn, str(library / "dive" / "GX020042.MP4"))
    assert db.stage_statuses(conn)[0]["counts"] == "2 videos (2 recordings)"
    for v in ingest.added(conn):
        ingest.remove(conn, v["path"])
    assert db.stage_statuses(conn)[0]["state"] == "not started"


def test_ingest_pages_and_sources_unchanged(library, tmp_path):
    before = digest(library)
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(library)]))
    client.post("/projects", data={"name": "p"})

    page = client.get("/p/p/ingest").text  # opens at video_roots
    assert "GX020042.MP4" in page and "also in" in page and "cannot read" in page
    assert "dive/" in page and "._GX" not in page

    reef = str(library / "copy" / "Site-25-08-29-1230-Diver-c2.MP4")  # first copy found
    assert reef in page
    outside = str(tmp_path / "elsewhere.MP4")  # not found by the scan: must be ignored
    r = client.post("/p/p/ingest/add", data={"folder": str(library), "paths": [reef, outside]},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/p/p/ingest?")
    assert "1 video (1 recording)" in client.get("/p/p").text

    client.post("/p/p/ingest/remove", data={"folder": str(library), "path": reef})
    assert "not started" in client.get("/p/p").text
    assert digest(library) == before  # SPEC constraint 4: source videos are read-only


def test_ingest_folder_box_and_errors(library, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(tmp_path / "gone")]))
    client.post("/projects", data={"name": "p"})
    assert "Paste a folder path" in client.get("/p/p/ingest").text
    assert "Folder not found" in client.get("/p/p/ingest?folder=/no/such/folder").text
    page = client.get("/p/p/ingest?" + urlencode({"folder": str(library / "dive")})).text
    assert "recording GX0042, chapter 2" in page
    r = client.post("/p/p/ingest/add", data={"folder": str(library), "paths": ["x"]},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403  # the host and origin rule covers ingest too (D14)
