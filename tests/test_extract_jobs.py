"""S4 bite 2: extraction and playback copies on the job module (D32): Pause / Resume of an
extraction through the pages, one job at a time across stages, playback cannot pause and
Cancel ends it. Synthetic videos made by ffmpeg at test time; no real footage."""

from app.stages import extract
from tests import grab_hooks
from tests.test_app_skeleton import client_for, write_config
from tests.test_clips import wait_for
from tests.test_ingest import digest, make_video


def project(tmp_path, seconds=10):
    lib = tmp_path / "lib"
    make_video(lib / "a.MP4", seconds=seconds, size="160x120")
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(lib)]))
    client.post("/projects", data={"name": "p"})
    client.post("/p/p/ingest/add", data={"folder": str(lib), "paths": [str(lib / "a.MP4")]})
    client.post("/p/p/extract/filter", data={"blur": "0", "dup": "0"})
    return client, lib, data / "projects" / "p"


def job(client):
    return client.get("/p/p/extract/status").json()["job"]


def test_extraction_pause_and_resume_decode_each_frame_once(tmp_path, monkeypatch):
    client, lib, folder = project(tmp_path)
    before = digest(lib)
    monkeypatch.setattr(extract, "GRAB_HOOK", "tests.grab_hooks:slow")
    client.post("/p/p/extract/start", data={"every": "1", "cap": "300"})   # 10 frames
    assert wait_for(lambda: job(client).get("done", 0) >= 2, 30)
    j = job(client)
    assert j["kind"] == "extract" and f"/jobs/{j['id']}/pause" in client.get("/p/p/extract").text
    client.post(f"/jobs/{j['id']}/pause", data={"back": "/p/p/extract"})
    assert wait_for(lambda: job(client)["state"] == "paused", 30)
    n = grab_hooks.calls(folder)
    assert 2 <= n < 10 and len(list((folder / "frames").glob("*.jpg"))) == n
    page = client.get("/p/p/extract").text
    assert "Extraction paused at" in page and f"/jobs/{j['id']}/resume" in page
    client.post(f"/jobs/{j['id']}/resume", data={"back": "/p/p/extract"})
    assert wait_for(lambda: job(client)["state"] == "done", 60)
    assert grab_hooks.calls(folder) == 10                # none decoded twice, none skipped
    assert client.get("/p/p/extract/status").json()["counts"]["kept"] == 10
    assert job(client)["message"] == f"{10 - n} new frames extracted"
    assert digest(lib) == before                         # SPEC constraint 4


def test_one_job_at_a_time_across_stages(tmp_path, monkeypatch):
    client, _, folder = project(tmp_path)
    monkeypatch.setattr(extract, "GRAB_HOOK", "tests.grab_hooks:slow")
    client.post("/p/p/extract/start", data={"every": "2", "cap": "300"})
    assert wait_for(lambda: job(client).get("state") == "running", 30)
    client.post("/p/p/clips/prepare", data={"v": ""})    # a playback copy waits behind it
    jc = client.app.state.jobctl
    from app import config
    jobs = jc(config.load(tmp_path / "config.local.toml")).list()
    copy = next(j for j in jobs if j["kind"] == "playback")
    assert copy["state"] == "queued"
    # one more Extract while it runs is refused, not queued twice
    r = client.post("/p/p/extract/start", data={"every": "2", "cap": "300"})
    assert "already queued or running" in r.text
    assert wait_for(lambda: job(client)["state"] == "done", 60)
    code = copy["params"]["code"]
    # wait for the job's own state: the files are ready a moment before the row says done
    # (QE PR 36: waiting for "ready" raced the runner)
    assert wait_for(lambda: client.get("/p/p/clips/status").json()[code].get("state") == "done", 60)
    assert client.get("/p/p/clips/status").json()[code]["ready"]
    ext = job(client)
    done = jc(config.load(tmp_path / "config.local.toml")).status(copy["id"])
    assert done["started"] >= ext["ended"]


def test_a_worker_error_names_its_cause(tmp_path):
    client, lib, folder = project(tmp_path)
    (lib / "a.MP4").rename(lib / "gone.MP4")             # the video went missing (SSD out)
    client.post("/p/p/clips/prepare", data={"v": ""})
    st = client.get("/p/p/clips/status").json()
    code = next(iter(st))
    assert wait_for(lambda: client.get("/p/p/clips/status").json()[code].get("state") == "error", 30)
    msg = client.get("/p/p/clips/status").json()[code]["message"]
    assert msg.startswith("Video not found:")            # the worker's own message, not a code
    assert "Video not found" in client.get("/jobs").text


def test_filters_applied_while_paused_hold_after_resume(tmp_path, monkeypatch):
    """QE PR 34: the resumed worker applies the values saved now, not those of Extract time."""
    client, _, folder = project(tmp_path)                 # filters 0 / 0 at Extract time
    monkeypatch.setattr(extract, "GRAB_HOOK", "tests.grab_hooks:slow")
    client.post("/p/p/extract/start", data={"every": "1", "cap": "300"})   # 10 frames
    assert wait_for(lambda: job(client).get("done", 0) >= 2, 30)
    client.post(f"/jobs/{job(client)['id']}/pause", data={"back": "/p/p/extract"})
    assert wait_for(lambda: job(client)["state"] == "paused", 30)
    r = client.post("/p/p/extract/filter", data={"blur": "20", "dup": "0"})
    assert r.status_code == 200 and "error=" not in str(r.url)
    client.post(f"/jobs/{job(client)['id']}/resume", data={"back": "/p/p/extract"})
    assert wait_for(lambda: job(client)["state"] == "done", 60)
    counts = client.get("/p/p/extract/status").json()["counts"]
    assert (counts["kept"], counts["dropped_blur"]) == (8, 2)   # 20 % of 10, as saved
