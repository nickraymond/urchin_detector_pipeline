"""S8 bite 4a: the train job kind on the job module, end to end with real worker processes and
a tiny fake trainer (tests/fake_trainer.py; no torch): a checkpoint and a metrics line per
epoch, Pause / Resume without a repeated or skipped step, Cancel keeps checkpoints."""

import ast
import json
from pathlib import Path

import pytest

from app.jobctl import JobControl
from app.train import job as train_job
from app.train import models
from tests.test_jobctl import REPO, wait_for, wait_state


@pytest.fixture
def jc(tmp_path, monkeypatch):
    monkeypatch.setitem(train_job.TRAINERS, "fake", "tests.fake_trainer:FakeTrainer")
    c = JobControl(tmp_path / "jobs")
    c.register(train_job.train_kind())
    yield c
    c.close()


def new_model(tmp_path, epochs=4, steps=5, **train):
    return models.create(tmp_path / "proj", "fake-tiny-64-c2", {
        "arch": {"kind": "detector", "family": "fake"},
        "train": {"epochs": epochs, "steps": steps, "seed": 1, **train}})


def steps(mdir) -> list[tuple[int, int]]:
    p = mdir / "steps.log"
    return [tuple(map(int, l.split()[:2])) for l in p.read_text().splitlines()] if p.exists() else []


def pids(mdir) -> set[str]:
    return {l.split()[2] for l in (mdir / "steps.log").read_text().splitlines()}


def submit(jc, mdir):
    return jc.submit("train", {"model_dir": str(mdir), "trainer": "fake"}, project="proj")


def every_step_once(epochs, n):
    return [(e, s) for e in range(epochs) for s in range(n)]


def test_runs_to_done_with_a_checkpoint_and_a_metrics_line_per_epoch(jc, tmp_path):
    mdir = new_model(tmp_path)
    assert mdir.name == "m1-fake-tiny-64-c2"
    jid = submit(jc, mdir)
    job = wait_state(jc, jid, "done", "error")
    assert job["state"] == "done" and (job["done"], job["total"]) == (4, 4), job
    rows = models.read_metrics(mdir)
    assert [r["epoch"] for r in rows] == [0, 1, 2, 3] and rows[-1]["value"] == 20
    assert all("seconds" in r and "loss" in r for r in rows)
    for e in (1, 2, 3, 4):
        assert (mdir / "checkpoints" / f"epoch_{e}_step_0.json").is_file()
    assert json.loads((mdir / "checkpoints" / "last.json").read_text()) == \
        {"epoch": 4, "step": 0, "value": 20}
    assert steps(mdir) == every_step_once(4, 5)
    # the run records what made it (spec stage 8): code version, device, libraries, limits
    (rec,) = models.read_recipe(mdir)["train"]["recorded"]
    assert rec["from_epoch"] == 0 and len(rec["code"]["commit"]) >= 7 and rec["device"]
    assert rec["libraries"] == {"fake_trainer": "1"} and rec["python"] and "limits" in rec
    assert job["summary"]["items"] == 4 and '"name": "epoch 4"' in (jc.root / f"{jid}.log").read_text()
    assert (mdir / "unloads.log").read_text().count("\n") == 1


def test_pause_mid_epoch_then_resume_repeats_and_skips_no_step(jc, tmp_path):
    mdir = new_model(tmp_path, epochs=3, steps=6, delay=0.1)
    jid = submit(jc, mdir)
    wait_for(lambda: len(steps(mdir)) >= 8)            # inside epoch 1
    jc.pause(jid)
    job = wait_state(jc, jid, "paused", "done", "error")
    assert job["state"] == "paused"
    ck = json.loads((mdir / "checkpoints" / "last.json").read_text())
    assert (ck["epoch"], ck["step"]) != (2, 0) and ck["value"] == len(steps(mdir))   # saved
    assert [r["epoch"] for r in models.read_metrics(mdir)] == list(range(ck["epoch"]))
    assert (mdir / "unloads.log").is_file()             # unloaded: the worker is gone
    first = models.read_recipe(mdir)["train"]["eval_scale"]
    jc.resume(jid)
    job = wait_state(jc, jid, "done", "error")
    assert job["state"] == "done" and job["done"] == 3
    assert steps(mdir) == every_step_once(3, 6) and len(pids(mdir)) == 2
    rows = models.read_metrics(mdir)
    assert [r["epoch"] for r in rows] == [0, 1, 2] and rows[-1]["value"] == 18
    rec = models.read_recipe(mdir)["train"]["recorded"]
    assert len(rec) == 2 and (rec[1]["from_epoch"], rec[1]["from_step"]) == (ck["epoch"], ck["step"])
    # D90: the eval factor is fixed at the first start; the resumed worker made a new trainer but kept it
    train = models.read_recipe(mdir)["train"]
    assert train["eval_overlap"] == 7 and train["eval_scale"] == first     # the first worker's, not the second's
    assert first["started_pid"] in pids(mdir) and len(pids(mdir)) == 2


def test_cancel_keeps_checkpoints_and_metrics(jc, tmp_path):
    mdir = new_model(tmp_path, epochs=5, steps=4, delay=0.1)
    jid = submit(jc, mdir)
    wait_for(lambda: len(models.read_metrics(mdir)) >= 1)
    jc.cancel(jid)
    assert wait_state(jc, jid, "cancelled")["stop_reason"] == "cancelled"
    done = len(models.read_metrics(mdir))
    assert 1 <= done < 5
    last = json.loads((mdir / "checkpoints" / "last.json").read_text())
    assert last["epoch"] in (done - 1, done)      # a whole file, at most one epoch behind
    assert not list((mdir / "checkpoints").glob(".*.tmp"))


def test_a_crash_between_the_metrics_line_and_the_checkpoint_repeats_no_line(jc, tmp_path):
    """The worker dies hard inside epoch 1. A new job on the same model continues from the
    last checkpoint; a metrics line written without its checkpoint is cut back, not doubled."""
    mdir = new_model(tmp_path, epochs=3, steps=4, crash_after="1:2")
    jid = submit(jc, mdir)
    job = wait_state(jc, jid, "error", "done")
    assert job["state"] == "error" and "exit code 3" in job["stop_reason"]
    assert [r["epoch"] for r in models.read_metrics(mdir)] == [0]
    # as if the app had died after epoch 1's metrics line but before its checkpoint
    with open(mdir / "train.jsonl", "a") as f:
        f.write(json.dumps({"epoch": 1, "loss": 9}) + "\n" + '{"epoch": 2, "lo')   # + a torn line
    jid2 = submit(jc, mdir)
    assert wait_state(jc, jid2, "done", "error")["state"] == "done"
    rows = models.read_metrics(mdir)
    assert [r["epoch"] for r in rows] == [0, 1, 2] and rows[1]["loss"] != 9
    # epoch 1's first two steps were lost with the crashed worker and are redone once
    assert steps(mdir) == every_step_once(1, 4) + [(1, 0), (1, 1)] + \
        [(1, s) for s in range(4)] + [(2, s) for s in range(4)]
    assert rows[-1]["value"] == 12


def test_progress_comes_from_the_model_folder_and_unknown_trainers_are_refused(jc, tmp_path):
    mdir = new_model(tmp_path, epochs=2, steps=1)
    assert train_job.progress({"model_dir": str(mdir)}) == (0, 2)
    jid = submit(jc, mdir)
    wait_state(jc, jid, "done")
    assert train_job.progress({"model_dir": str(mdir)}) == (2, 2)
    bad = jc.submit("train", {"model_dir": str(mdir), "trainer": "os:system"})
    assert wait_state(jc, bad, "error")["state"] == "error"
    with pytest.raises(ValueError, match="No trainer"):
        train_job.prepare({"trainer": "os:system"})


def test_model_ids_are_numbered_and_never_reused(tmp_path):
    a = models.create(tmp_path, "yolox-nano-conv-256-c5", {"arch": {}, "train": {"epochs": 1}})
    b = models.create(tmp_path, "yolox-tiny-conv-256-c5", {"arch": {}, "train": {"epochs": 1}})
    assert (a.name, b.name) == ("m1-yolox-nano-conv-256-c5", "m2-yolox-tiny-conv-256-c5")
    assert models.list_ids(tmp_path) == [a.name, b.name]
    assert models.model_dir(tmp_path, a.name) == a
    for bad in ("../x", "m9-nope", "", "m1"):
        with pytest.raises(models.ModelError):
            models.model_dir(tmp_path, bad)
    with pytest.raises(models.ModelError, match="Bad architecture key"):
        models.create(tmp_path, "Bad Key", {})


def test_the_job_side_of_training_never_imports_torch():
    """The worker loop, the kind and the models folder are tested without torch; only a real
    trainer module (bite 4b) may import it."""
    for name in ("worker.py", "job.py", "models.py"):
        tree = ast.parse((Path(REPO) / "app" / "train" / name).read_text())
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] in ("torch", "torchvision") for n in names), name


def test_a_paused_estimate_stops_without_a_result(tmp_path, monkeypatch):
    """QE PR 87 nit 1: an estimate answers Pause (it stops; Resume runs it again)."""
    from app.train import worker
    events = []
    monkeypatch.setattr(worker.jw, "emit", lambda event, **more: events.append(event))
    mdir = tmp_path / "models" / ".estimate"
    (mdir / "checkpoints").mkdir(parents=True)
    recipe = {"arch": {}, "train": {"epochs": 4, "steps": 3}}

    class Ctl:
        def __init__(self, answers):
            self.answers = iter(answers)

        def pause_requested(self):
            return next(self.answers)
    job = {"trainer": "tests.fake_trainer:FakeTrainer"}
    worker.estimate(job, mdir, recipe, Ctl([True]))              # Pause came while loading
    worker.estimate(job, mdir, recipe, Ctl([False, True]))       # ... or while timing
    assert events == ["paused", "paused"] and not (mdir / "estimate.json").exists()
    worker.estimate(job, mdir, recipe, Ctl([False, False]))
    assert events[2:] == ["item", "done"] and (mdir / "estimate.json").is_file()
