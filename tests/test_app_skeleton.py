"""S2 bite 1a: settings errors, projects, dashboard and project pages. Temporary folders only."""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import config, db, project
from app.web import server


def write_config(tmp_path, data_dir, roots=None):
    path = tmp_path / "config.local.toml"
    roots = roots if roots is not None else [str(tmp_path)]
    path.write_text(f'video_roots = {json.dumps(roots)}\ndata_dir = "{data_dir}"\n')
    return path


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return data, TestClient(server.create_app(write_config(tmp_path, data)))


# --- settings ---

def test_missing_config_file_gives_fix(tmp_path):
    cfg = config.load(tmp_path / "nope.toml")
    assert not cfg.ok and "cp config.example.toml config.local.toml" in cfg.problems[0]


def test_bad_toml_and_missing_data_dir(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("data_dir = [unclosed")
    assert "Cannot read" in config.load(bad).problems[0]
    bad.write_text('video_roots = []\n')
    assert "has no data_dir" in config.load(bad).problems[0]


def test_data_dir_not_found_is_never_created(tmp_path):
    missing = tmp_path / "unplugged_ssd" / "data"
    cfg = config.load(write_config(tmp_path, missing))
    assert "Data folder not found" in cfg.problems[0]
    assert not missing.exists()


def test_missing_video_root_is_a_warning_only(tmp_path):
    cfg = config.load(write_config(tmp_path, tmp_path, roots=[str(tmp_path / "gone")]))
    assert cfg.ok and "Video folder not found" in cfg.warnings[0]


# --- projects ---

@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden", "with space", "x" * 65])
def test_bad_project_names_refused(tmp_path, bad):
    with pytest.raises(project.ProjectError):
        project.create(tmp_path, bad)
    assert not (tmp_path / "projects").exists() or not any((tmp_path / "projects").iterdir())


def test_create_open_list(tmp_path):
    folder = project.create(tmp_path, "reef_1")
    info = json.loads((folder / "project.json").read_text())
    assert info["name"] == "reef_1" and info["settings"]["frame_cap"] == 300
    tables = {r[0] for r in sqlite3.connect(folder / "state.sqlite").execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"videos", "frames", "stage_status", "review_actions"} <= tables
    assert project.open_project(tmp_path, "reef_1")["folder"] == folder
    assert project.list_projects(tmp_path) == ["reef_1"]
    with pytest.raises(project.ProjectError, match="already exists"):
        project.create(tmp_path, "reef_1")


def test_damaged_project_json_gives_fix(tmp_path):
    folder = project.create(tmp_path, "p")
    (folder / "project.json").write_text("{half")
    with pytest.raises(project.ProjectError, match="cannot be read"):
        project.open_project(tmp_path, "p")


def test_stage_status_order_and_update(tmp_path):
    conn = db.connect(tmp_path / "s.sqlite")
    assert [s["stage"] for s in db.stage_statuses(conn)] == [s for s, _ in db.STAGES]
    db.set_stage(conn, "ingest", "done", "2 videos")
    assert db.stage_statuses(conn)[0]["counts"] == "2 videos"
    with pytest.raises(ValueError):
        db.set_stage(conn, "ingest", "finished")


# --- pages ---

def test_dashboard_shows_setup_problem_not_a_crash(tmp_path):
    client = TestClient(server.create_app(tmp_path / "missing.toml"))
    r = client.get("/")
    assert r.status_code == 200 and "Setup problem" in r.text and "Traceback" not in r.text
    assert client.post("/projects", data={"name": "x"}).status_code == 503


def test_create_project_through_page(setup):
    data, client = setup
    r = client.post("/projects", data={"name": "reef_1"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/p/reef_1"
    page = client.get("/p/reef_1").text
    for _, label in db.STAGES:
        assert label in page
    assert "reef_1" in client.get("/").text
    assert (data / "projects" / "reef_1" / "state.sqlite").is_file()


def test_bad_name_and_unknown_project_pages(setup):
    data, client = setup
    r = client.post("/projects", data={"name": "../escape"})
    assert r.status_code == 400 and "Bad project name" in r.text
    assert not (data / "escape").exists()
    r = client.get("/p/nobody")
    assert r.status_code == 404 and "No project named" in r.text


def test_pages_load_nothing_from_the_internet(setup):
    _, client = setup
    client.post("/projects", data={"name": "p"})
    for url in ("/", "/p/p", "/static/style.css"):
        text = client.get(url).text
        assert "http://" not in text.replace("http://127.0.0.1", "") and "https://" not in text


def test_server_binds_localhost_only():
    assert server.HOST == "127.0.0.1"
