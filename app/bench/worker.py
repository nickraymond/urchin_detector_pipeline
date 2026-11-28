"""The `bench` job's worker (S7 bite 2b): `python -m app.bench.worker <job.json>`.

Checks the card again (reviewed, unchanged since submit, its unit allowed, its model files unchanged), saves
it with the run, opens the board, loads the models, then times each window and model, appending one line
per window to timing.jsonl. Pause is taken after a window; a new worker skips the windows already on disk
(after loading the models again). A stop rule (BoardError) ends the job with what to do next; the finished
windows stay."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from app import config
from app.bench import card as cards
from app.bench import run as runs
from app.bench.board import (BAD_ANSWER, LOAD_FAILED, LOW_MEMORY, NO_ANSWER, NOT_BUILT, NOT_LOADED, READBACK,
                             RESTORE_FAILED, BoardError, Model, open_board)
from app.bench.units import Refused
from app.jobctl import worker as jw

HINTS = {   # D84: what to do after each stop rule
    NO_ANSWER: "the board did not answer: check its cable and power, then run the card again",
    READBACK: "the model area's read-back differs: unless the message says the built-in restore already put it back "
              "(the N6), put that board back from the backup (RESTORE.md) before "
              "anything else, and note it in the rig log",
    LOAD_FAILED: "the load failed: check the rig log for this change and undo it before the next run",
    NOT_LOADED: "the model was not on the board when it was timed: run the card again; if it repeats, report it",
    BAD_ANSWER: "the bench agent printed something that is not its answer: read agent.log in the run folder and "
                "report it; nothing more runs until it is understood",
    LOW_MEMORY: "the Pi had under 50 MB free (D89): find what uses its memory (another process?) before running again",
    NOT_BUILT: "this board's route is not built yet: use the fake board or wait for its bite",
    RESTORE_FAILED: "the N6's model area is NOT known to be the backup's: run nothing more on the rig; put ROMFS0 "
                    "back by hand (RESTORE.md, \"N6: restore a partition\"), replugging the N6 if it shows neither "
                    "its serial port nor its bootloader, and read it back before the next card",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write(path, body: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True))
    os.replace(tmp, path)


def board_settings(c: dict, unit) -> dict:
    """What the card fixes on the board itself: a real Pi runs the card's thread count (D83, D88)."""
    return {"threads": c.get("threads", 1)} if c["board"] == "pi" and not unit.fake else {}


def _close(board) -> None:
    """End the agent's process on the Pi (a PiBoard); the fake board has nothing to close."""
    close = getattr(board, "close", None)
    if close:
        close()


def main() -> None:
    job = jw.load_job()
    ctl = jw.listen()
    cfg = config.load(job["config"])
    if not cfg.ok:
        raise RuntimeError("the settings file has a problem: " + "; ".join(cfg.problems))
    data_dir, folder = cfg.data_dir, runs.run_dir(cfg.data_dir, job["run_id"])
    c = cards.load(data_dir, job["card"])
    if not c["reviewed"] or c["sha256"] != job["card_sha256"]:
        raise RuntimeError(f"card {job['card']} changed or lost its review after Run was pressed; write a new card")
    unit = cfg.bench.unit(c["unit"])                      # refused here too if [bench] changed meanwhile
    for m in c["models"]:
        if cards.sha256_file(m["path"]) != m["sha256"]:
            raise RuntimeError(f"model file {m['path']} changed since card {c['number']} was written; write a new card")
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "card.json").is_file():
        _write(folder / "card.json", c)
    run = json.loads((folder / "run.json").read_text()) if (folder / "run.json").is_file() else {
        "card": c["number"], "card_sha256": c["sha256"], "unit": unit.name, "board": c["board"],
        "fake": unit.fake, "models": c["models"], "windows": c["windows"], "n": c["n"], "started": _now()}
    run["job_id"] = job.get("job_id")             # the page reads the job's state (a Cancel ends the worker at once)
    try:
        board = open_board(unit, c["board"], data_dir, bench=cfg.bench, log_dir=folder,
                           **board_settings(c, unit), **job.get("fake", {}))
        run["board_info"] = board.info()
        loaded = board.load_several([Model(m["name"], m["path"]) for m in c["models"]])
        run["route"] = loaded.get("route", "")
        run.update(outcome="running", ended=None)
        _write(folder / "run.json", run)
        done = {(w.get("window"), w.get("model")) for w in runs.windows_done(folder)}
        total, i = c["windows"] * len(c["models"]), len(done)
        for window in range(1, c["windows"] + 1):
            if ctl.pause_requested():                     # Pause after a window (D84)
                run["outcome"] = "paused"
                _write(folder / "run.json", run)
                _close(board)
                jw.emit("paused")
                return
            for m in c["models"]:
                if (window, m["name"]) in done:
                    continue
                t = board.time(m["name"], c["n"])
                with open(folder / "timing.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"window": window, "model": m["name"], "at": _now(), **t}) + "\n")
                i += 1
                jw.emit("item", i=i, n=total, name=f"window {window}, {m['name']}",
                        seconds=round(t["ms_mean"] * t["n"] / 1000, 3))
    except BoardError as e:
        _close(locals().get("board"))
        run.update(outcome=f"stopped: {e.stop}", ended=_now(), error=str(e),
                   restores=getattr(locals().get("board"), "restores", []))     # the N6's built-in restore (bite 4b)
        _write(folder / "run.json", run)
        raise RuntimeError(f"stop rule {e.stop}: {e}. Next: {HINTS.get(e.stop, 'see the job log')}. "
                           "The windows already timed are kept.") from None
    except Refused as e:                  # e.g. the alias now resolves to a field unit (QE PR 112 nit 2)
        _close(locals().get("board"))
        run.update(outcome="stopped: refused", ended=_now(), error=str(e))
        _write(folder / "run.json", run)
        raise RuntimeError(f"refused: {e} Nothing more was done on the unit; the windows already timed are kept.") from None
    _close(board)
    run.update(outcome="done", ended=_now())
    _write(folder / "run.json", run)
    s = runs.summary(folder)["models"]
    jw.emit("done", message="; ".join(f"{k}: median {v['median_ms']:.2f} ms over {v['windows']} windows"
                                      for k, v in s.items() if v["median_ms"] is not None) or "done")


if __name__ == "__main__":
    jw.run(main)
