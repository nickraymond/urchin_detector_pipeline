"""Follow-ups from QE PR 105 (nits 1-4) and PR 113 (nits 3-4)."""

import json
from datetime import datetime

import pytest

from app.bench import run as runs
from app.bench.agent import bench_agent as ag
from tests.test_bench_run import new_card, site  # noqa: F401


def test_two_runs_in_one_second_get_their_own_folders(tmp_path):
    at = datetime(2026, 11, 27, 19, 0, 0)
    ids = [runs.reserve_run_id(tmp_path, 4, now=at) for _ in range(3)]
    assert ids == ["20261127-190000-card4", "20261127-190000-card4-2", "20261127-190000-card4-3"]
    assert all(runs.run_dir(tmp_path, i).is_dir() for i in ids)


def test_summary_takes_no_memory_figure_as_none(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "run.json").write_text(json.dumps({"card": 1}))
    lines = [{"window": w, "model": "m", "n": 3, "ms_mean": 2.0 + w, "ms_min": 1.0, "ms_max": 9.0,
              "mem_after_load": None} for w in (1, 2, 3)]
    (folder / "timing.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    m = runs.summary(folder)["models"]["m"]
    assert m["median_ms"] == 4.0 and m["mem_after_load"] is None
    lines[1]["mem_after_load"] = 400_000_000
    (folder / "timing.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    assert runs.summary(folder)["models"]["m"]["mem_after_load"] == 400_000_000


def test_a_card_file_that_cannot_be_read_is_a_page_not_a_500(site):  # noqa: F811
    cfg, data = site
    from fastapi.testclient import TestClient
    from app.web import server
    n = new_card(data, cfg.bench)
    (data / "bench" / "cards" / f"{n}.json").write_text("{ not json")
    app = server.create_app(cfg.path)
    try:
        r = TestClient(app, base_url="http://127.0.0.1:8000").get(f"/bench/cards/{n}")
        assert r.status_code == 422 and "cannot be read" in r.text and f"bench/cards/{n}.json" in r.text
    finally:
        app.state.close_jobs()


def test_the_install_also_needs_50_mb_free(tmp_path, monkeypatch):
    folder = tmp_path / "agent"
    folder.mkdir()
    req = folder / "requirements-litert-2.2.0.txt"
    req.write_text("# pins\n")
    monkeypatch.setattr(ag, "proc_kib", lambda path, key: 20 * 1024 * 1024 if key == "MemAvailable" else None)
    with pytest.raises(ag.Stop) as e:
        ag.Agent(folder, "fake").op_install({"runtime": "litert"})
    assert e.value.stop == ag.LOW_MEMORY and "an install" in str(e.value)
    assert not (folder / "venv-litert-2.2.0").exists()
