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
                             BOARD_RESET, INPUT_DIFFERS, RESTORE_FAILED, SCRIPT_FAILED, BoardError, Model, open_board)
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
    BOARD_RESET: "the N6 reset itself during the timing (its port dropped or it showed its bootloader); the job put "
                 "ROMFS0 back. The message names the last stage that answered: report it; the next card isolates that "
                 "stage (a vendor model already in /rom needs no write)",
    SCRIPT_FAILED: "a statement on the N6 failed (the message names the stage and quotes the board): report it. A "
                   "failing stage 2 (ml.Model) means the model does not load on this firmware only when the board's "
                   "words are not a NameError or ImportError of the script's own names (that would be the stages "
                   "losing their state, not the model)",
    INPUT_DIFFERS: "the N6 timed another input than the laptop's (D83a): report it; the run's numbers are not used",
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
    """What the card fixes on the board itself: a real Pi runs the card's thread count (D83, D88) under the
    runtime its model files name (.tflite: LiteRT, .onnx: ONNX Runtime; D85)."""
    if c["board"] != "pi" or unit.fake:
        return {}
    return {"threads": c.get("threads", 1), "runtime": c.get("runtime", "litert")}


def _close(board) -> None:
    """End the agent's process on the Pi (a PiBoard); the fake board has nothing to close."""
    close = getattr(board, "close", None)
    if close:
        close()


def _end_restore(board, run: dict):
    """Lead call (lead 2, PR 123): every N6 card ends with ROMFS0 put back from the backup and read back, pass or
    fail, then a hard reset of the booted board so its own main.py runs again (lead 2, after PR 123): the board is
    left as found. Returns the BoardError of either step, or None."""
    if board is None or getattr(board, "kind", "") != "n6":
        return None
    if not getattr(board, "reached", True) and not getattr(board, "image_on_board", False):
        return None                       # the board never answered: nothing to restore or reset (QE PR 126 nit 2)
    try:
        if getattr(board, "image_on_board", False):
            board.restore()
        run["finish"] = board.finish()
        return None
    except BoardError as e:
        if e.stop != RESTORE_FAILED:
            e = BoardError(e.stop, f"the end-of-card hard reset failed: {e}")
        return e
    finally:
        run["restores"] = list(getattr(board, "restores", []))


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
        if c.get("restore_only"):                         # the N6 rehearsal card: the restore and nothing else
            r = board.restore()
            failed = _end_restore(board, run)             # restored already: the hard reset only
            if failed:
                raise failed
            run.update(outcome="done", ended=_now(), restores=list(board.restores))
            _write(folder / "run.json", run)
            _close(board)
            jw.emit("done", message=f"ROMFS0 restored from the backup: read back identical "
                                    f"({r['readback_sha256'][:16]}…), /rom lists {len(r['rom'])} entries")
            return
        if c.get("rom_models"):                           # models already in the N6's /rom: no write (after card 6)
            loaded = board.use_rom(c["rom_models"])
        else:
            loaded = board.load_several([Model(m["name"], m["path"]) for m in c["models"]])
        timed = c["models"] or [{"name": n, "rom": True} for n in c.get("rom_models", [])]
        run["route"] = loaded.get("route", "")
        run.update(outcome="running", ended=None)
        _write(folder / "run.json", run)
        done = {(w.get("window"), w.get("model")) for w in runs.windows_done(folder)}
        total, i = c["windows"] * len(timed), len(done)
        for window in range(1, c["windows"] + 1):
            if ctl.pause_requested():                     # Pause after a window (D84)
                failed = _end_restore(board, run)         # an N6 is left as found; Resume loads again
                if failed:
                    raise failed
                run["outcome"] = "paused"
                _write(folder / "run.json", run)
                _close(board)
                jw.emit("paused")
                return
            for m in timed:
                if (window, m["name"]) in done:
                    continue
                t = board.time(m["name"], c["n"])
                with open(folder / "timing.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"window": window, "model": m["name"], "at": _now(), **t}) + "\n")
                i += 1
                jw.emit("item", i=i, n=total, name=f"window {window}, {m['name']}",
                        seconds=round(t["ms_mean"] * t["n"] / 1000, 3))
    except BoardError as e:
        board = locals().get("board")
        if e.stop != RESTORE_FAILED:                      # a failed restore is not tried twice
            failed = _end_restore(board, run)
            if failed:
                e = BoardError(failed.stop, f"{e} — and then the end of the card did not finish: {failed}")
        _close(board)
        if e.detail.get("stage_failed"):                  # the N6's stages: which one, and the last that answered
            run["stage"] = {k: e.detail[k] for k in ("stage_failed", "stage_name", "last_stage") if k in e.detail}
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
    except Exception as e:                # QE PR 123: any other exit (a full disk, a bad answer) still ends restored (D82c)
        board = locals().get("board")
        failed = _end_restore(board, run)
        _close(board)
        note = (f" The end of the card did not finish: {failed}. Next: {HINTS.get(failed.stop, 'see the job log')}."
                if failed else " ROMFS0 was put back from the backup." if run.get("restores") else "")
        run.update(outcome=f"stopped: {failed.stop if failed else 'error'}", ended=_now(),
                   error=f"{type(e).__name__}: {e}.{note}")
        try:
            _write(folder / "run.json", run)
        except OSError:
            pass                          # the disk may be the fault; the job's error still says it all
        raise RuntimeError(f"the run stopped on an error the board did not report: {type(e).__name__}: {e}.{note} "
                           "The windows already timed are kept.") from None
    failed = _end_restore(board, run)
    _close(board)
    if failed:
        run.update(outcome=f"stopped: {failed.stop}", ended=_now(), error=str(failed))
        _write(folder / "run.json", run)
        raise RuntimeError(f"stop rule {failed.stop}: every window was timed, but the end of the card (restore, then "
                           f"hard reset) did not finish: {failed}. Next: {HINTS.get(failed.stop, 'see the job log')}.") from None
    run.update(outcome="done", ended=_now())
    _write(folder / "run.json", run)
    s = runs.summary(folder)["models"]
    jw.emit("done", message="; ".join(f"{k}: median {v['median_ms']:.2f} ms over {v['windows']} windows"
                                      for k, v in s.items() if v["median_ms"] is not None) or "done")


if __name__ == "__main__":
    jw.run(main)
