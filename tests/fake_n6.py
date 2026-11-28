"""A fake N6 for the bench agent's tests (S7 bite 4b): `python fake_n6.py <state dir> mpremote|dfu-util <args>`.

It answers the command lines the agent uses, as RESTORE.md and ml/README.md show them, over a small fake ROMFS0 held
in <state>/romfs0.bin. State files: `mode` (app | dfu | dead), `faults.json`, `calls.log` (one JSON argv per call).
Faults: write_fails (n writes fail), corrupt_write (n writes flip a byte), dead_unless_backup (a boot with anything
but backup.bin in ROMFS0 leaves the board answering nothing, neither serial nor DFU), time_fails, hide (a /rom name).
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib  # noqa: E402

from app.bench import romfs  # noqa: E402
from app.bench.agent.bench_agent import synthetic_bytes  # noqa: E402

SERIAL = "FAKE0N6SERIAL"


def dfu_list(path, serial):
    return "".join(f'Found DFU: [37c5:9206] ver=0102, devnum=13, cfg=1, intf=0, path="{path}", alt={a}, name="{n}", '
                   f'serial="{serial}"\n' for a, n in ((3, "ROMFS0"), (2, "FILESYSTEM"), (1, "FIRMWARE"), (0, "BOOTLOADER")))


class State:
    def __init__(self, d):
        self.d = Path(d)

    def get(self, name, default=""):
        p = self.d / name
        return p.read_text() if p.is_file() else default

    def put(self, name, text):
        (self.d / name).write_text(text)

    @property
    def faults(self):
        return json.loads(self.get("faults.json", "{}"))

    def take(self, fault):
        f = self.faults
        if f.get(fault, 0) > 0:
            f[fault] -= 1
            self.put("faults.json", json.dumps(f))
            return True
        return False


def mpremote(s, args):
    port, verb = args[1], args[2]
    code = args[3] if len(args) > 3 else ""
    if "OpenMV_Camera" in port:                       # the AE3: answers who it is, nothing else is asked of it
        print("NB:AE3")
        return 0
    if s.get("mode", "app") != "app":
        print(f"mpremote: failed to access {port} (it may be in use by another program)")
        return 1
    if verb in ("soft-reset", "reset"):
        return 0
    if "machine.bootloader()" in code:
        s.put("mode", "dfu")
        print("Traceback (most recent call last):\nOSError: [Errno 5] Input/output error")
        return 1
    entries, _ = romfs.decode((s.d / "romfs0.bin").read_bytes())
    rom = {e.name: len(e.data) for e in entries if e.name != s.faults.get("hide")}
    if "os.listdir('/rom')" in code:
        print("NB:" + json.dumps(rom))
    elif "ml.Model" in code:
        name = re.search(r"ml\.Model\('/rom/([^']+)'\)", code).group(1)
        n = int(re.search(r"for _ in range\((\d+)\):\n    t = time", code).group(1))
        if name not in rom or s.faults.get("time_fails"):
            print("Traceback (most recent call last):\nOSError: Failed to load network")
            return 1
        image = synthetic_bytes(8, 8, 3) + (b"!" if s.faults.get("input_differs") else b"")
        print("NB:" + json.dumps({"us_total": 10_600 * n, "us_min": 10_500, "us_max": 10_700, "input_shape": [1, 8, 8, 3],
                                  "input_sha256": hashlib.sha256(image).hexdigest(), "input_bytes": len(image),
                                  "mem_before_load": 25_000_000, "mem_after_load": 24_000_000, "mem_after": 24_000_000}))
    elif "firmware" in code:
        print("NB:" + json.dumps({"board_type": "N6", "firmware": "fake N6 5.0.1", "machine": "fake N6",
                                  "mem_free": 25_000_000}))
    elif "omv.board_type()" in code:
        print("NB:N6")
    else:
        print("fake N6: unknown script")
        return 1
    return 0


def dfu_util(s, args):
    serial = s.get("serial", SERIAL)
    if s.get("mode", "app") != "dfu" or ("-S" in args and args[args.index("-S") + 1] != serial):
        print("dfu-util: No DFU capable USB device available")
        return 74
    if "-l" in args:
        print(dfu_list("1-1.3", serial) + (dfu_list("1-1.4", "SOMEONE_ELSE") if s.faults.get("two_devices") else ""))
        return 0
    alt = int(args[args.index("-a") + 1])
    part = s.d / "romfs0.bin"
    size = len(part.read_bytes())
    if "-D" in args:
        if alt != 3:
            s.put(f"alt{alt}_written", "1")               # the tests check this never happens
            return 0
        if s.take("write_fails"):
            print("dfu-util: Error during download")
            return 74
        data = Path(args[args.index("-D") + 1]).read_bytes()
        new = bytearray(data + b"\xff" * (size - len(data)))
        if s.take("corrupt_write"):
            new[100] ^= 0xFF
        part.write_bytes(bytes(new))
        print("Download done.")
        return 0
    if "-U" in args:
        out = Path(args[args.index("-U") + 1])
        if out.exists():
            print(f"dfu-util: Cannot open file {out} for writing")
            return 74
        out.write_bytes(part.read_bytes() + b"\x00" * 4 if alt == 3 else b"\x00" * 64)
        if "-R" in args:
            ok = not s.faults.get("dead_unless_backup") or part.read_bytes() == (s.d / "backup.bin").read_bytes()
            s.put("mode", "app" if ok else "dead")
            return 251                                    # RESTORE.md: exit 251 after -R is normal
        return 0
    return 1


def main(argv):
    s = State(argv[1])
    with open(s.d / "calls.log", "a") as f:
        f.write(json.dumps(argv[2:]) + "\n")
    return (mpremote if argv[2] == "mpremote" else dfu_util)(s, argv[3:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
