"""The `pi` board (S7 bite 3, D81, D85): the bench Pi runs the model itself, under the runtime being measured,
through the bench agent. The model file is copied to the agent's folder (D82, Pi route); the agent checks its
sha256 and times it (D83). LiteRT first, on the same 8-bit TFLite file S6 makes.

`Board.load_several` writes the rig log around the copy (D81). Copying the agent itself onto the Pi is a rig change
too: `install_agent()` logs it with its undo. Tests and builders use a LocalTransport (the agent as a child process
on the laptop, the "fake" runtime or LiteRT from the conversion environment); never a real unit.
"""

from __future__ import annotations

from pathlib import Path

from app.bench import riglog
from app.bench.board import Board, Model
from app.bench.agent.bench_agent import INSTALLS
from app.bench.transport import REMOTE_FOLDER, Transport
from app.bench.units import Unit

AGENT_UNDO = f"delete the agent's folder ~/{REMOTE_FOLDER} on the Pi"


class PiBoard(Board):
    kind = "pi"

    def __init__(self, unit: Unit, data_dir, transport: Transport, threads: int = 1, install: str | None = None):
        """install: the runtime whose environment the agent runs in ("litert", D89); None runs the agent with the
        transport's own Python (tests)."""
        self.unit, self.data_dir, self.transport, self.threads = unit, Path(data_dir), transport, threads
        self.install, self.runtime_record = install, None
        self.hello: dict | None = None

    def install_agent(self) -> None:
        """Copy the agent and its pin files into its folder: a rig change, logged with its undo."""
        with riglog.change(self.data_dir, self.unit.name, self.kind, "copy the bench agent", AGENT_UNDO,
                           fake=self.unit.fake):
            self.transport.install()

    def prepare(self) -> dict:
        """Before the first connection (D89): the agent copied if it changed; the runtime's environment built if it
        is not ready (its own rig change, with its undo); then the agent is started from that environment."""
        if not self.transport.agent_current():
            self.install_agent()
        spec = INSTALLS[self.install]
        boot = self.transport.with_python(self.transport.system_python)
        try:
            boot.start()
            status = boot.request("install_status", runtime=self.install)
            if status["ready"]:
                record = status["installed"]
            else:
                with riglog.change(self.data_dir, self.unit.name, self.kind,
                                   f"install the {self.install} runtime into ~/{REMOTE_FOLDER}/{spec['venv']}",
                                   f"delete ~/{REMOTE_FOLDER}/{spec['venv']} on the Pi", fake=self.unit.fake,
                                   requirements=spec["requirements"]):
                    record = boot.request("install", runtime=self.install)["installed"]
        finally:
            boot.close()
        self.transport = self.transport.with_python(self.transport.venv_python(spec["venv"]))
        self.runtime_record = record
        return record

    def connect(self) -> dict:
        if self.hello is None:
            if self.install and self.runtime_record is None:
                self.prepare()
            self.hello = {**self.transport.start(), "runtime_install": self.runtime_record}
        return self.hello

    def info(self) -> dict:
        self.connect()
        return {**self.transport.request("info", kind="pi"), "agent": self.hello}

    def load_several(self, models: list[Model]) -> dict:
        self.connect()            # the agent copy and the runtime install are their own rig changes, before the load's
        return super().load_several(models)

    def _load(self, models: list[Model], files: dict) -> dict:
        self.connect()
        for m in models:
            self.transport.put(Path(m.path), m.name)
        return self.transport.request("load", kind="pi", files=files, threads=self.threads)

    def time(self, name: str, n: int) -> dict:
        self.connect()
        return self.transport.request("time", kind="pi", name=name, n=n)

    def reset(self) -> None:
        self.connect()
        self.transport.request("reset", kind="pi")

    def close(self) -> None:
        self.transport.close()
        self.hello = None
