"""A tiny fake trainer for the train job's tests (no torch): each "step" adds 1 to a number
and appends "<epoch> <step> <pid>" to steps.log, so a test can prove that no step ran twice or
was left out across Pause / Resume / a crash. The checkpoint is a small JSON file.

Recipe train settings it reads: epochs, steps (per epoch), delay (s per step), crash_after
(a "epoch:step" at which the first worker to get there exits hard, once)."""

import json
import os
import time
from pathlib import Path


class FakeTrainer:
    def __init__(self, model_dir, recipe, job):
        self.dir, self.t = Path(model_dir), recipe["train"]
        self.value = 0

    def describe(self):
        return {"device": "none (fake)", "libraries": {"fake_trainer": "1"}}

    def load(self):
        p = self.dir / "checkpoints" / "last.json"
        if not p.is_file():
            return 0, 0
        ck = json.loads(p.read_text())
        self.value = ck["value"]
        return ck["epoch"], ck["step"]

    def run_epoch(self, epoch, start_step, stop):
        for step in range(start_step, self.t.get("steps", 3)):
            if stop():
                return {"done": False, "step": step}
            mark = self.dir / ".crashed"
            if self.t.get("crash_after") == f"{epoch}:{step}" and not mark.exists():
                mark.write_text("x")
                os._exit(3)
            time.sleep(self.t.get("delay", 0))
            self.value += 1
            with open(self.dir / "steps.log", "a") as f:
                f.write(f"{epoch} {step} {os.getpid()}\n")
        return {"done": True, "metrics": {"loss": round(1 / (1 + self.value), 4),
                                          "value": self.value}}

    def save(self, epoch, step):
        p = self.dir / "checkpoints" / "last.json"
        tmp = p.with_name(".last.json.tmp")
        tmp.write_text(json.dumps({"epoch": epoch, "step": step, "value": self.value}))
        os.replace(tmp, p)
        (self.dir / "checkpoints" / f"epoch_{epoch}_step_{step}.json").write_text(
            json.dumps({"value": self.value}))

    def unload(self):
        with open(self.dir / "unloads.log", "a") as f:
            f.write(f"{os.getpid()}\n")
