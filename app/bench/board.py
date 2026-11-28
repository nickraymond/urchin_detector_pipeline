"""The `Board` interface (D81) and the fake board every test, builder and QE uses (SPEC constraint 9).

A board is one of three kinds: `pi` (the bench Pi itself), `n6`, `ae3`. Its five operations:

    info()                      kind, firmware, free memory, model area size
    load(model)                 put one model file on the board (a rig change: logged before and after)
    load_several(models)        put several on the board together (one model area image / one process)
    time(name, n)               one timing window (D83): 2 warm-ups, then n inferences; returns at least n,
                                warmups, ms_mean, ms_min, ms_max, mem_before_load, mem_after_load, mem_after
                                (free memory in bytes; the bench job reads these, QE PR 108)
    reset()                     back to a known state (a soft reset; no rig change)

A failure is a BoardError whose `stop` names the test card's stop rule (D84). The real kinds come with
bites 3 (pi, `pi_board.py`), 4 (n6, `n6_board.py`) and 5 (ae3); `open_board` refuses the AE3 until then.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path

from app.bench import riglog
from app.bench.units import Refused, Unit

KINDS = ("pi", "n6", "ae3")
WARMUPS = 2                               # D83
NO_ANSWER, READBACK, LOAD_FAILED, NOT_LOADED = "no_answer", "readback_differs", "load_failed", "not_loaded"
# from the bench agent (S7 bite 3b): a line that is not JSON, free memory under the floor, a route not built yet
BAD_ANSWER, LOW_MEMORY, NOT_BUILT = "bad_answer", "low_memory", "not_built"


# the N6 (S7 bite 4b): the built-in restore of ROMFS0 from the backup did not verify
RESTORE_FAILED = "restore_failed"
INPUT_DIFFERS = "input_differs"           # D83a: the N6's input image is not the laptop's
# the N6's timing in stages (after card 6): the board reset itself mid-stage; a board-side statement failed
BOARD_RESET, SCRIPT_FAILED = "board_reset", "script_failed"


class BoardError(Exception):
    def __init__(self, stop: str, message: str, detail: dict | None = None):
        super().__init__(message)
        self.stop, self.detail = stop, dict(detail or {})


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
                "n6": "write the backup's ROMFS0 partition back (DFU alt 3: the job's own restore, or RESTORE.md "
                      "\"N6: restore a partition\"), then delete n6_*.img from ~/urchin_bench/models on the Pi",
                "ae3": "delete the file from /flash, or write the backup's ROMFS0 partition back"}[self.kind]

    def _load(self, models: list[Model], files: dict) -> dict:
        raise NotImplementedError


class FakeBoard(Board):
    """No hardware. Timings are seeded and repeatable; memory and failures are set by the test.

    ms:      base ms per inference per model name (default 10.0); windows vary by up to `jitter` of it
    fail:    {"load" | "time" | "info" | "reset": stop} to raise BoardError with that stop rule
    fail_at: the time() call number (1, 2, ...) at which "time" fails, when set; earlier calls work
    delay_s: seconds each time() call takes (tests of Pause and Cancel)
    """

    def __init__(self, unit: Unit, kind: str, data_dir, ms: dict | None = None, jitter: float = 0.05,
                 mem_free: int = 4_090_000, model_bytes: int = 1_000_000, fail: dict | None = None, seed: int = 0,
                 fail_at: int = 0, delay_s: float = 0.0):
        if not unit.fake:
            raise ValueError("FakeBoard runs only on the fake unit")
        if kind not in unit.kinds:
            raise ValueError(f"no {kind} board on unit {unit.name}")
        self.unit, self.kind, self.data_dir = unit, kind, Path(data_dir)
        self.ms, self.jitter, self.mem_free, self.model_bytes = ms or {}, jitter, mem_free, model_bytes
        self.fail, self.rng = dict(fail or {}), random.Random(seed)
        self.fail_at, self.delay_s, self.timed = fail_at, delay_s, 0
        self.loaded: dict[str, str] = {}
        self.calls: list[str] = []
        self.image_on_board, self.restores = False, []    # the N6's model area (bite 4b), pretended
        self.reached = False                               # the board answered once (as N6Board)

    def _maybe_fail(self, op: str) -> None:
        self.calls.append(op)
        if op == "time":
            self.timed += 1
            if self.fail_at and self.timed < self.fail_at:
                return
        if op in self.fail and self.fail[op] == "plain_error":     # tests: a fault that is not a BoardError
            raise RuntimeError(f"fake {self.kind} board: {op} raised a plain error as asked")
        if op in self.fail:
            raise BoardError(self.fail[op], f"fake {self.kind} board: {op} failed ({self.fail[op]}) as asked")

    def info(self) -> dict:
        self._maybe_fail("info")
        self.reached = True
        return {"kind": self.kind, "firmware": "fake 1.0", "mem_free": self.mem_free,
                "model_area_bytes": 0 if self.kind == "pi" else 0x01800000, "fake": True}

    def _load(self, models, files) -> dict:
        self._maybe_fail("load")
        self.loaded = dict(files)
        self.image_on_board = self.kind == "n6"
        return {"loaded": sorted(files), "route": "fake", "mem_free": self.mem_free - self.model_bytes * len(files)}

    def time(self, name: str, n: int) -> dict:
        self._maybe_fail("time")
        if name not in self.loaded:
            raise BoardError(NOT_LOADED, f"{name} is not loaded on the fake {self.kind} board")
        if n < 1:
            raise ValueError("a window needs at least one inference")
        if self.delay_s:
            time.sleep(self.delay_s)
        base = float(self.ms.get(name, 10.0))
        each = [base * (1 + self.rng.uniform(-self.jitter, self.jitter)) for _ in range(n)]
        after_load = self.mem_free - self.model_bytes * len(self.loaded)
        return {"n": n, "warmups": WARMUPS, "ms_mean": sum(each) / n, "ms_min": min(each), "ms_max": max(each),
                "mem_before_load": self.mem_free, "mem_after_load": after_load, "mem_after": after_load}

    def reset(self) -> None:
        self._maybe_fail("reset")
        self.loaded = {}

    def finish(self) -> dict:
        """The fake N6's end-of-card hard reset (fail={"finish": ...} to make it fail)."""
        if self.kind != "n6":
            raise ValueError("only the N6 ends a card with a hard reset")
        self._maybe_fail("finish")
        return {"hard_reset": True, "fake": True}

    def use_rom(self, names: list[str]) -> dict:
        """The fake N6's models already in /rom: timed with no write (fail={"load": ...} as for a load)."""
        self._maybe_fail("load")
        self.reached = True
        self.loaded = {n: "rom" for n in names}
        return {"loaded": sorted(names), "route": "already in /rom (no write)", "fake": True}

    def restore(self) -> dict:
        """The fake N6's "ROMFS0 back from the backup" (fail={"restore": ...} to make it fail), logged as the real one."""
        if self.kind != "n6":
            raise ValueError("only the N6 has a model area to restore")
        with riglog.change(self.data_dir, self.unit.name, self.kind, "restore ROMFS0 from the backup (DFU alt 3)",
                           "none needed: this writes the backup's own ROMFS0 back", fake=True):
            try:
                self._maybe_fail("restore")
            except BoardError as e:                    # as N6Board.restore: any failure is restore_failed
                raise BoardError(RESTORE_FAILED, str(e)) from None
        self.image_on_board, self.loaded = False, {}
        done = {"restored": True, "readback_sha256": "f" * 64, "rom": {"fake.tflite": 1}, "fake": True}
        self.restores.append(done)
        return done


def open_board(unit: Unit, kind: str, data_dir, bench=None, log_dir=None, threads: int = 1, **fake) -> Board:
    """The board to run on. The fake unit gets a FakeBoard. A real unit's Pi gets a PiBoard over ssh (S7 bite 3b),
    built only from a unit `bench.unit()` returns again here; its cameras are refused until bites 4-5."""
    if kind not in KINDS:
        raise ValueError(f"unknown board kind {kind!r}; kinds are {', '.join(KINDS)}")
    if unit.fake:
        return FakeBoard(unit, kind, data_dir, **fake)
    if fake:
        raise Refused("fake settings are for the fake unit only")
    if kind in ("pi", "n6"):
        if bench is None or bench.unit(unit.name) != unit:
            raise Refused(f"{unit.name} was not given by [bench]; a real unit is only used as [bench] lists it")
        if kind not in unit.kinds:
            raise Refused(f"[bench] lists no {kind} on unit {unit.name}")
        from app.bench.transport import SshTransport
        log = Path(log_dir or data_dir) / "agent.log"
        if kind == "n6":                  # S7 bite 4b: the agent runs with the system python3 (no runtime needed)
            from app.bench.n6_board import N6Board
            if not bench.n6_dfu_serial:
                raise Refused("the N6 route needs the bench N6's DFU serial: set [bench] n6_dfu_serial in "
                              "config.local.toml (the rig backup's cameras/n6/dfu_list.txt); nothing is written to an "
                              "unpinned N6")
            return N6Board(unit, data_dir, SshTransport(unit, bench.field_units, log=log), work_dir=log_dir or data_dir,
                           dfu_serial=bench.n6_dfu_serial)
        from app.bench.pi_board import PiBoard
        return PiBoard(unit, data_dir, SshTransport(unit, bench.field_units, log=log), threads=threads, install="litert")
    raise Refused(f"the {kind} route on real units is not built yet (S7 bite 5); use the fake board")
