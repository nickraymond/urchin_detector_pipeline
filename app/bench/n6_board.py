"""The `n6` board (S7 bite 4b; D82, D82a, D83, D84): the N6 on the bench Pi's USB, through the bench agent.

The N6 loads models only from its model area, ROMFS0 (24 MiB), and an image replaces the whole partition. So a load
builds one image on the laptop: the board's own entries from the rig backup, byte for byte and in their order, then
ours (D82a, `romfs.with_ours`). The agent writes it with `dfu-util -a 3 -D` (alt 3 only; alt 0 is never written),
reads it back (compare capped at the image's length), boots the board and checks the `/rom` listing.

The way back is built in: any stop rule from a load that had started writing, from a timing window or from a reset
runs `restore()` before the error reaches the job: the backup's partition (the backup file's first 25,165,824 bytes;
its last 4 are dfu-util's read padding) written to alt 3, read back whole, booted, `/rom` equal to the backup's 10
entries. The restore is its own rig-log change. A restore that does not verify is the `restore_failed` stop rule.

Builders and QE run this against the fake N6 (tests/fake_n6.py): fake mpremote and dfu-util over a small partition.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.bench import riglog, romfs
from app.bench.agent.bench_agent import file_name
from app.bench.board import LOAD_FAILED, NOT_LOADED, RESTORE_FAILED, BoardError, Model
from app.bench.pi_board import PiBoard
from app.bench.transport import Transport
from app.bench.units import Unit

PARTITION = 0x01800000            # OMV_ROMFS_PART0_LENGTH (boards/OPENMV_N6/board_config.h), 25,165,824 bytes
DFU_PAD = 4                       # dfu-util 0.11's read appends 4 bytes past the partition (RESTORE.md, gap 4)
# The rig backup of 2026-11-27 (never in git): its ROMFS0 file, read twice on the Pi and identical
# (cameras/n6/dfu/n6_dfu_run.log: 25,165,828 bytes, sha256 dbffa6d2…).
BACKUP = Path("~/NereusData/urchin_detector_pipeline/rig_backup/2026-11-27/cameras/n6/dfu/n6_ROMFS0.bin").expanduser()
BACKUP_SHA256 = "dbffa6d2ee0474c7229c6e65e0af0339dd9da5d756abda995805b5551f7268d3"
IMAGE_NAME, BACKUP_NAME = "n6_romfs0_ours.img", "n6_romfs0_backup.img"     # in the agent's models/ folder
RESTORE_UNDO = "none needed: this writes the backup's own ROMFS0 back (the state before the bench work)"


def rom_name(name: str) -> str:
    """Our file's name in /rom: the card's model name (S6's `<arch key>`) made safe, ending in .tflite (as
    OpenMV's own models do, so `default_align` gives it 32)."""
    base = file_name(name)
    return base if base.endswith(".tflite") else base + ".tflite"


class Backup:
    """The backup's partition, checked before anything is built or sent."""

    def __init__(self, path: Path, sha256: str, partition: int):
        raw = Path(path).read_bytes() if Path(path).is_file() else b""
        if not raw:
            raise BoardError(LOAD_FAILED, f"the N6's ROMFS0 backup is not at {path}: nothing was sent to the rig")
        if len(raw) != partition + DFU_PAD or hashlib.sha256(raw).hexdigest() != sha256:
            raise BoardError(LOAD_FAILED, f"the N6's ROMFS0 backup at {path} is {len(raw):,} bytes with sha256 "
                                          f"{hashlib.sha256(raw).hexdigest()[:16]}…, not the recorded "
                                          f"{partition + DFU_PAD:,} bytes and {sha256[:16]}…: nothing was sent")
        self.partition = raw[:partition]
        self.entries, self.length = romfs.decode(self.partition)
        if romfs.encode(self.entries) != self.partition[:self.length]:     # QE PR 119 nit 1, before anything is built
            raise BoardError(LOAD_FAILED, "re-encoding the backup's ROMFS0 entries does not give the backup's image "
                                          "back byte for byte (D82a's known-good check): nothing was sent")
        self.sha256 = hashlib.sha256(self.partition).hexdigest()
        self.rom = {e.name: len(e.data) for e in self.entries}


class N6Board(PiBoard):
    kind = "n6"

    def __init__(self, unit: Unit, data_dir, transport: Transport, work_dir=None, backup: Path = BACKUP,
                 backup_sha256: str = BACKUP_SHA256, partition: int = PARTITION):
        super().__init__(unit, data_dir, transport, install=None)
        self.work_dir = Path(work_dir or data_dir)
        self.backup_path, self.backup_sha256, self.partition = Path(backup), backup_sha256, partition
        self.rom_names: dict[str, str] = {}
        self.image_on_board = False      # True from the moment a write may have started until a restore verifies
        self.restores: list[dict] = []

    def connect(self) -> dict:
        if self.hello is None:
            if not self.transport.agent_current():
                self.install_agent()                 # its own rig change (D88); the agent needs no runtime here
            self.hello = self.transport.start()
        return self.hello

    def info(self) -> dict:
        self.connect()
        return {**self.transport.request("info", kind="n6"), "agent": self.hello}

    def _load(self, models: list[Model], files: dict) -> dict:
        backup = Backup(self.backup_path, self.backup_sha256, self.partition)
        ours = [romfs.Entry(rom_name(m.name), Path(m.path).read_bytes(), 32) for m in models]
        try:
            image = romfs.with_ours(backup.entries, ours, self.partition)
        except romfs.RomfsError as e:
            raise BoardError(LOAD_FAILED, f"our ROMFS0 image cannot be built: {e}. Nothing was sent") from None
        self.work_dir.mkdir(parents=True, exist_ok=True)
        local_image, local_backup = self.work_dir / IMAGE_NAME, self.work_dir / BACKUP_NAME
        local_image.write_bytes(image)
        local_backup.write_bytes(backup.partition)
        self.connect()
        self.transport.put(local_backup, BACKUP_NAME)          # the way back is on the Pi before the first write
        self.transport.put(local_image, IMAGE_NAME)
        rom = {**backup.rom, **{e.name: len(e.data) for e in ours}}
        self.rom_names = {m.name: rom_name(m.name) for m in models}
        self._backup = backup
        try:
            done = self.transport.request("load", kind="n6", image=IMAGE_NAME,
                                          image_sha256=hashlib.sha256(image).hexdigest(), image_len=len(image),
                                          backup=BACKUP_NAME, backup_sha256=backup.sha256, rom=rom,
                                          ours=sorted(self.rom_names))
        except BoardError as e:
            if e.detail.get("written", True):               # unknown (the agent ended, a timeout): restore anyway
                self.image_on_board = True
                self._restore_after(e)
            raise
        self.image_on_board = True
        return {**done, "image_sha256": hashlib.sha256(image).hexdigest(), "image_bytes": len(image),
                "rom_names": self.rom_names}

    def time(self, name: str, n: int) -> dict:
        self.connect()
        if name not in self.rom_names:
            raise BoardError(NOT_LOADED, f"{name} is not in the N6's model area: load it first")
        try:
            return self.transport.request("time", kind="n6", rom_name=self.rom_names[name], n=n)
        except BoardError as e:
            self._restore_after(e)
            raise

    def reset(self) -> None:
        self.connect()
        try:
            self.transport.request("reset", kind="n6")
        except BoardError as e:
            self._restore_after(e)
            raise

    def _restore_after(self, e: BoardError) -> None:
        """Run the restore for a stop rule; the error then carries its result (or becomes restore_failed)."""
        if not self.image_on_board:
            return
        try:
            done = self.restore()
        except BoardError as r:
            e.stop = RESTORE_FAILED
            e.args = (f"{e} — and then the built-in restore of ROMFS0 did not verify: {r}",)
            return
        e.args = (f"{e} — ROMFS0 was then put back from the backup: read back identical "
                  f"({done['readback_sha256'][:16]}…), /rom lists the backup's {len(done['rom'])} entries",)

    def restore(self) -> dict:
        """The backup's ROMFS0 back on the board, verified by read-back and the /rom listing (a rig change)."""
        backup = getattr(self, "_backup", None) or Backup(self.backup_path, self.backup_sha256, self.partition)
        with riglog.change(self.data_dir, self.unit.name, self.kind, "restore ROMFS0 from the backup (DFU alt 3)",
                           RESTORE_UNDO, fake=self.unit.fake, sha256={BACKUP_NAME: backup.sha256}):
            try:
                if self.transport.proc is None or self.transport.proc.poll() is not None:
                    self.hello = None                        # the agent ended (a fault): start it again for the restore
                    self.connect()
                done = self.transport.request("restore", kind="n6", backup=BACKUP_NAME, backup_sha256=backup.sha256,
                                              rom=backup.rom)
            except BoardError as r:
                raise BoardError(RESTORE_FAILED, str(r)) from None
        self.image_on_board, self.rom_names = False, {}
        self.restores.append(done)
        return done
