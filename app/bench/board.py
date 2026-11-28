"""The `Board` interface (D81) and the fake board every test, builder and QE uses (SPEC constraint 9).

A board is one of three kinds: `pi` (the bench Pi itself), `n6`, `ae3`. Its five operations:

    info()                      kind, firmware, free memory, model area size
    load(model)                 put one model file on the board (a rig change: logged before and after)
    load_several(models)        put several on the board together (one model area image / one process)
    time(name, n)               one timing window (D83): 2 warm-ups, then n inferences; ms and memory
    reset()                     back to a known state (a soft reset; no rig change)

A failure is a BoardError whose `stop` names the test card's stop rule (D84). The real kinds come with
bites 3 (pi), 4 (n6) and 5 (ae3); `open_board` refuses them until then.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from app.bench import riglog
from app.bench.units import Refused, Unit

KINDS = ("pi", "n6", "ae3")
WARMUPS = 2                               # D83
NO_ANSWER, READBACK, LOAD_FAILED, NOT_LOADED = "no_answer", "readback_differs", "load_failed", "not_loaded"


class BoardError(Exception):
    def __init__(self, stop: str, message: str):
        super().__init__(message)
        self.stop = stop


@dataclass(frozen=True)
class Model:
    name: str
    path: Path

    def sha256(self) -> str:
        return hashlib.sha256(Path(self.path).read_bytes()).hexdigest()


class Board:
    """What the bench job calls. Subclasses do the board work; this class writes the rig log."""

    kind: str
    unit: Unit
    data_dir: Path

    def info(self) -> dict:
        raise NotImplementedError

    def load(self, model: Model) -> dict:
        return self.load_several([model])

    def load_several(self, models: list[Model]) -> dict:
        names = ", ".join(m.name for m in models)
        files = {m.name: m.sha256() for m in models}
        with riglog.change(self.data_dir, self.unit.name, self.kind, f"load {names}",
                           self._undo(), fake=self.unit.fake, sha256=files):
            return self._load(models, files)

    def time(self, name: str, n: int) -> dict:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def _undo(self) -> str:
        return {"pi": "delete the model files from the agent's folder on the Pi",
                "n6": "write the backup's ROMFS0 partition back (DFU alt 3)",
                "ae3": "delete the file from /flash, or write the backup's ROMFS0 partition back"}[self.kind]

    def _load(self, models: list[Model], files: dict) -> dict:
        raise NotImplementedError


class FakeBoard(Board):
    """No hardware. Timings are seeded and repeatable; memory and failures are set by the test.

    ms:      base ms per inference per model name (default 10.0); windows vary by up to `jitter` of it
    fail:    {"load" | "time" | "info" | "reset": stop} to raise BoardError with that stop rule
    """

    def __init__(self, unit: Unit, kind: str, data_dir, ms: dict | None = None, jitter: float = 0.05,
                 mem_free: int = 4_090_000, model_bytes: int = 1_000_000, fail: dict | None = None, seed: int = 0):
        if not unit.fake:
            raise ValueError("FakeBoard runs only on the fake unit")
        if kind not in unit.kinds:
            raise ValueError(f"no {kind} board on unit {unit.name}")
        self.unit, self.kind, self.data_dir = unit, kind, Path(data_dir)
        self.ms, self.jitter, self.mem_free, self.model_bytes = ms or {}, jitter, mem_free, model_bytes
        self.fail, self.rng = dict(fail or {}), random.Random(seed)
        self.loaded: dict[str, str] = {}
        self.calls: list[str] = []

    def _maybe_fail(self, op: str) -> None:
        self.calls.append(op)
        if op in self.fail:
            raise BoardError(self.fail[op], f"fake {self.kind} board: {op} failed ({self.fail[op]}) as asked")

    def info(self) -> dict:
        self._maybe_fail("info")
        return {"kind": self.kind, "firmware": "fake 1.0", "mem_free": self.mem_free,
                "model_area_bytes": 0 if self.kind == "pi" else 0x01800000, "fake": True}

    def _load(self, models, files) -> dict:
        self._maybe_fail("load")
        self.loaded = dict(files)
        return {"loaded": sorted(files), "route": "fake", "mem_free": self.mem_free - self.model_bytes * len(files)}

    def time(self, name: str, n: int) -> dict:
        self._maybe_fail("time")
        if name not in self.loaded:
            raise BoardError(NOT_LOADED, f"{name} is not loaded on the fake {self.kind} board")
        if n < 1:
            raise ValueError("a window needs at least one inference")
        base = float(self.ms.get(name, 10.0))
        each = [base * (1 + self.rng.uniform(-self.jitter, self.jitter)) for _ in range(n)]
        after_load = self.mem_free - self.model_bytes * len(self.loaded)
        return {"n": n, "warmups": WARMUPS, "ms_mean": sum(each) / n, "ms_min": min(each), "ms_max": max(each),
                "mem_before_load": self.mem_free, "mem_after_load": after_load, "mem_after": after_load}

    def reset(self) -> None:
        self._maybe_fail("reset")
        self.loaded = {}


def open_board(unit: Unit, kind: str, data_dir, **fake) -> Board:
    """The board to run on. Only the fake unit has a board in this bite; real units are refused."""
    if kind not in KINDS:
        raise ValueError(f"unknown board kind {kind!r}; kinds are {', '.join(KINDS)}")
    if unit.fake:
        return FakeBoard(unit, kind, data_dir, **fake)
    raise Refused(f"the {kind} route on real units is not built yet (S7 bites 3-5); use the fake board")
