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


def test_same_video_from_two_folders_added_once(library, tmp_path):
    """QE finding 1 on PR 4: D13 holds across scans, not only within one."""
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(library)]))
    client.post("/projects", data={"name": "p"})
    name = "Site-25-08-29-1230-Diver-c2.MP4"
    for sub in ("site", "copy"):
        folder = str(library / sub)
        client.post("/p/p/ingest/add", data={"folder": folder, "paths": [f"{folder}/{name}"]})
    assert "1 video (1 recording)" in client.get("/p/p").text
    page = client.get("/p/p/ingest?" + urlencode({"folder": str(library / "copy")})).text
    assert "added (other folder)" in page and 'name="paths"' not in page


def test_big_folder_refused_with_hint(library, tmp_path, monkeypatch):
    """QE finding 2 on PR 4: a huge folder must not hang the page."""
    monkeypatch.setattr(ingest, "MAX_VIDEOS", 2)
    with pytest.raises(ingest.IngestError, match="Open one of its subfolders"):
        ingest.scan(library)
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(library)]))
    client.post("/projects", data={"name": "p"})
    page = client.get("/p/p/ingest").text
    assert "more than 2 at once" in page and "dive/" in page


def test_decode_timeout_names_ffmpeg(library, monkeypatch):
    """QE nit 3 on PR 4: a decode timeout must not blame ffprobe."""
    real_run = ingest.subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] == "ffmpeg":
            raise ingest.subprocess.TimeoutExpired(cmd, 60)
        return real_run(cmd, **kw)

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    monkeypatch.setattr(ingest, "_probe_cache", {})
    warning = ingest.probe(library / "dive" / "GX010042.MP4")["warning"]
    assert warning.startswith("ffmpeg cannot read")


def test_thumbnails_made_in_data_dir_and_served(library, tmp_path):
    """Nick's request on PR 4: a Finder-style thumbnail per video."""
    before = digest(library)
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(library / "dive")]))
    client.post("/projects", data={"name": "p"})
    page = client.get("/p/p/ingest").text
    code = ingest.path_code(library / "dive" / "GX010042.MP4")
    assert f'src="/thumbs/{code}.jpg"' in page
    thumb = data / "thumbs" / f"{code}.jpg"
    assert thumb.stat().st_size > 500
    r = client.get(f"/thumbs/{code}.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"  # a real JPEG
    for bad in ("zzzzzz", "..%2Fx", "abcdef"):
        assert client.get(f"/thumbs/{bad}.jpg").status_code == 404
    assert digest(library) == before  # nothing written beside the videos


def test_no_other_site_may_embed_responses(library, tmp_path):
    """QE nit 1 on PR 4: Cross-Origin-Resource-Policy on every response, images included."""
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(library / "dive")]))
    client.post("/projects", data={"name": "p"})
    client.get("/p/p/ingest")
    code = ingest.path_code(library / "dive" / "GX010042.MP4")
    for r in (client.get("/"), client.get(f"/thumbs/{code}.jpg"), client.get("/static/style.css"),
              client.get("/thumbs/zzzzzz.jpg"),
              client.post("/projects", data={"name": "q"}, headers={"Origin": "http://e.x"})):
        assert r.headers["cross-origin-resource-policy"] == "same-origin"


def test_thumbnail_rebuilt_when_video_changes(tmp_path):
    """QE nit 2 on PR 4: a replaced video at the same path gets a new thumbnail."""
    video = make_video(tmp_path / "v" / "a.MP4", seconds=1)
    thumbs = tmp_path / "thumbs"
    first = ingest.thumbnail(ingest.scan(tmp_path / "v")[0], thumbs).read_bytes()
    make_video(video, seconds=2, size="128x96")
    ingest._probe_cache.clear()
    second = ingest.thumbnail(ingest.scan(tmp_path / "v")[0], thumbs).read_bytes()
    assert first != second
