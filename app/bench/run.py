"""The `bench` job (S7 bite 2b; D83, D84): one reviewed test card -> windows -> <data_dir>/bench/<run id>/.

    card.json       the card and its review, as they were when the run started (saved with the run)
    run.json        unit, board, its info, models and sha256, route, start / end, the outcome
    timing.jsonl    one line per window and model: n, ms mean / min / max, memory readouts (append-only)

App side only (no board work): params at submit, progress from timing.jsonl, the summary the pages show.
The worker is app/bench/worker.py."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

from app import sidecars
from app.bench import card as cards
from app.bench import units
from app.jobctl import JobError, Kind


def run_dir(data_dir, run_id: str) -> Path:
    return Path(data_dir) / "bench" / run_id


def reserve_run_id(data_dir, number: int, now=None) -> str:
    """`<YYYYmmdd-HHMMSS>-card<n>`, with `-2`, `-3`, ... when that folder exists: the folder is made here, at once
    (mkdir fails if it exists), so two Runs in one second never share a folder (QE PR 105 nit 1)."""
    base = f"{(now or datetime.now()):%Y%m%d-%H%M%S}-card{number}"
    for k in range(1, 1000):
        run_id = base if k == 1 else f"{base}-{k}"
        try:
            run_dir(data_dir, run_id).mkdir(parents=True)
            return run_id
        except FileExistsError:
            continue
    raise JobError(f"no free run folder for card {number} at {base}")


def params(cfg, number: int, fake: dict | None = None) -> dict:
    """What the page submits. Refused unless the card is reviewed and its unit may be used. `fake` (tests
    only) sets the fake board's timings and failures; it is refused for a real unit."""
    try:
        c = cards.load(cfg.data_dir, number)
    except cards.CardError as e:
        raise JobError(str(e)) from None
    if not c["reviewed"]:
        raise JobError(f"Card {number} cannot be run: " + (
            "it changed after its review; write a new card." if c["review"] else
            "it is not reviewed yet (D84: someone other than its writer reviews it first)."))
    try:
        unit = cfg.bench.unit(c["unit"])
    except units.Refused as e:
        raise JobError(f"Card {number} cannot be run: {e}") from None
    if not unit.fake and not cards.has_restore_rule(c["stop_rules"]):
        raise JobError(f'Card {number} cannot be run: a card for a real unit keeps the stop rule '
                       f'"{cards.RESTORE_RULE}" (D87); write a new card.')
    if fake and not unit.fake:
        raise JobError("fake board settings are for the fake unit only")
    run_id = reserve_run_id(cfg.data_dir, number)
    return {"config": str(cfg.path), "data_dir": str(cfg.data_dir), "card": number,
            "card_sha256": c["sha256"], "run_id": run_id, "fake": fake or {}}


def windows_done(folder: Path) -> list[dict]:
    p = Path(folder) / "timing.jsonl"
    if not p.is_file():
        return []
    out = []
    for text in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(text))
        except ValueError:
            continue        # a line cut by a crash; that window is timed again
    return out


def progress(p: dict) -> tuple[int, int]:
    folder = run_dir(p["data_dir"], p["run_id"])
    try:
        c = json.loads((folder / "card.json").read_text()) if (folder / "card.json").is_file() \
            else cards.load(p["data_dir"], p["card"])
    except (cards.CardError, ValueError):
        return 0, 0
    keys = {(w.get("window"), w.get("model")) for w in windows_done(folder)}
    return len(keys), c["windows"] * len(c["models"])


def bench_kind() -> Kind:
    return Kind("bench", module="app.bench.worker", progress=progress, item="window")


def summary(folder: Path) -> dict:
    """Per model: windows timed, median of the window means, spread (lowest and highest window mean), lowest
    free memory after load. A model with fewer than 3 windows has no median (D83: never from one window)."""
    folder = Path(folder)
    run = json.loads((folder / "run.json").read_text()) if (folder / "run.json").is_file() else {}
    per: dict[str, list[dict]] = {}
    for w in windows_done(folder):
        per.setdefault(w["model"], []).append(w)
    models = {}
    for name, ws in per.items():
        means = [w["ms_mean"] for w in ws]
        models[name] = {"windows": len(ws), "median_ms": statistics.median(means) if len(ws) >= cards.MIN_WINDOWS else None,
                        "low_ms": min(means), "high_ms": max(means),
                        # None where the board gives no figure (the laptop's agent has no /proc): QE PR 105 nit 4
                        "mem_after_load": min((w["mem_after_load"] for w in ws if w.get("mem_after_load") is not None),
                                              default=None)}
    return {"run_id": folder.name, "run": run, "models": models}


def runs_for_card(data_dir, number: int) -> list[dict]:
    root = Path(data_dir) / "bench"
    found = []
    for p in sidecars.real(root.glob("*/run.json")) if root.is_dir() else []:
        try:
            if json.loads(p.read_text()).get("card") == number:
                found.append(summary(p.parent))
        except ValueError:
            continue
    return sorted(found, key=lambda s: s["run_id"], reverse=True)
