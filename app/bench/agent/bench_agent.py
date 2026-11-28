"""The bench agent (S7 bite 3, D81): runs ON the bench Pi, in its own folder, started by the laptop over ssh as
`python3 bench_agent.py <folder> [--runtime litert|fake]`. Plain Python 3 and its standard library; no import from
the app (it is copied alone). numpy and the runtime are imported only when a model is loaded.

Protocol: JSON lines. The laptop writes one request per line, {"id": n, "op": ..., ...}; the agent answers each
with one line, {"id": n, "ok": true, ...} or {"id": n, "ok": false, "stop": <stop rule>, "error": <words>}.
Ops: hello, info, load, time, reset, bye, install_status, install (D89); restore (n6). Model files are copied into <folder>/models/ by the laptop before
"load"; the agent only checks their sha256 (a mismatch is the "readback_differs" stop rule, D84).

Timing (D83): one seeded synthetic input at the model's input size; 2 warm-ups; time.perf_counter_ns() around
each invoke of the window; peak memory is the process's VmHWM. Threads are fixed per load and recorded. While a
window runs the agent writes {"id": n, "alive": true, "done": k} after the first inference that ends ALIVE_S
seconds or more after the last such line (D84: after each inference at the latest once 10 s have passed), so a slow model (300
inferences at a second each) is not taken for a board that stopped answering. `time` returns the fields every
Board.time returns (n, warmups, ms_mean, ms_min, ms_max, mem_before_load, mem_after_load, mem_after: free memory,
MemAvailable) plus vm_hwm.
The N6 (bite 4b, kind "n6"): found by asking it; load writes our ROMFS0 image with dfu-util alt 3 (no other alt is
ever written), reads it back, boots and lists /rom; restore does the same with the backup's partition; time runs
STAGES (one exec each) through mpremote. mpremote and dfu-util are the Pi's own (RESTORE.md); tests pass fake ones. The AE3 is
bite 5: asked for it, the agent says so.
"""

import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time

VERSION = "1"
WARMUPS = 2
ALIVE_S = 5.0          # an "alive" line this often during a window: D84 asks at least every 10 s; 60 s is for silence
SEED = 0
NO_ANSWER, READBACK, LOAD_FAILED, NOT_LOADED, NOT_BUILT, LOW_MEMORY = (
    "no_answer", "readback_differs", "load_failed", "not_loaded", "not_built", "low_memory")
INSTALL_FAILED = "install_failed"
MEM_FLOOR = 50 * 1024 * 1024   # D89: no load and no window when MemAvailable is under 50 MB


def file_name(name):
    """The file a model is copied to in <folder>/models/: its name with anything but letters, digits, '.', '_' and
    '-' replaced by '_' (the laptop uses the same rule; a name such as "fit/x/model.tflite" stays unique)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class Stop(Exception):
    def __init__(self, stop, message, **extra):
        super().__init__(message)
        self.stop, self.extra = stop, extra        # extra goes back in the answer (an N6 load says "written")


def proc_kib(path, key):
    """A "<key>: <n> kB" line of a /proc file, in bytes; None where there is no /proc (the laptop)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FakeRuntime:
    """No runtime and no numpy: invoke does nothing (or sleeps `ms`). For the laptop's tests of the protocol."""
    name = "fake"
    ms = 0.0

    def open(self, path, threads):
        return {"path": path}

    def invoke(self, model):
        if self.ms:
            time.sleep(self.ms / 1000)

    def describe(self, model):
        return {"input": None, "outputs": []}


class LiteRT:
    """LiteRT (ai-edge-litert, Apache-2.0; D85) on the 8-bit TFLite file S6 makes."""
    name = "litert"

    def open(self, path, threads):
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=path, num_threads=threads)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        rng = np.random.default_rng(SEED)
        dtype = np.dtype(inp["dtype"])
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            data = rng.integers(info.min, info.max, size=inp["shape"], endpoint=True, dtype=dtype)
        else:
            data = rng.uniform(0.0, 1.0, size=inp["shape"]).astype(dtype)
        it.set_tensor(inp["index"], data)
        return {"it": it, "inp": inp, "np": np}

    def invoke(self, model):
        model["it"].invoke()

    def describe(self, model):
        np = model["np"]
        shape = lambda d: {"shape": [int(v) for v in d["shape"]], "dtype": np.dtype(d["dtype"]).name}
        return {"input": shape(model["inp"]), "outputs": [shape(o) for o in model["it"].get_output_details()]}

    def version(self):
        from importlib import metadata
        return "ai-edge-litert " + metadata.version("ai-edge-litert")


class OnnxRuntime:
    """ONNX Runtime (onnxruntime, MIT; D85), the second Pi candidate, on an 8-bit QDQ ONNX file; the CPU provider,
    `threads` intra-op threads and one inter-op thread, so the count is the card's."""
    name = "onnxruntime"
    DTYPES = {"tensor(float)": "float32", "tensor(uint8)": "uint8", "tensor(int8)": "int8", "tensor(float16)": "float16"}

    def open(self, path, threads):
        import numpy as np
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        if inp.type not in self.DTYPES:
            raise ValueError(f"input type {inp.type} is not one this agent makes data for")
        dtype = np.dtype(self.DTYPES[inp.type])
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]      # a free dimension: batch 1
        rng = np.random.default_rng(SEED)
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            data = rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=dtype)
        else:
            data = rng.uniform(0.0, 1.0, size=shape).astype(dtype)
        return {"sess": sess, "feed": {inp.name: data}, "inp": inp, "shape": shape, "dtype": dtype.name}

    def invoke(self, model):
        model["sess"].run(None, model["feed"])

    def describe(self, model):
        return {"input": {"shape": model["shape"], "dtype": model["dtype"]},
                "outputs": [{"shape": [d if isinstance(d, int) else None for d in o.shape], "dtype": self.DTYPES.get(o.type, o.type)}
                            for o in model["sess"].get_outputs()]}

    def version(self):
        import onnxruntime as ort
        return "onnxruntime " + ort.__version__


RUNTIMES = {"fake": FakeRuntime, "litert": LiteRT, "onnxruntime": OnnxRuntime}

# D89: a runtime's own environment inside the agent's folder, from a hash-pinned file the laptop copies next to the agent
INSTALLS = {"litert": {"venv": "venv-litert-2.2.0", "requirements": "requirements-litert-2.2.0.txt"},
            "onnxruntime": {"venv": "venv-onnxruntime-1.29.0", "requirements": "requirements-onnxruntime-1.29.0.txt"}}
SUFFIX_RUNTIME = {".tflite": "litert", ".onnx": "onnxruntime"}    # a Pi card's runtime follows its model files
PIP_ARGS = ["install", "--no-cache-dir", "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
            "--require-hashes", "--no-input", "-r"]
FAKE_INSTALL = False            # tests only (--fake-install): make the environment without venv or pip


def head_tail(text, keep):
    """The start and the end of a long output (QE 10, card 6: the board's first lines were cut when only the tail was
    kept): `keep` characters of each, with the cut said in between."""
    if len(text) <= 2 * keep:
        return text
    return f"{text[:keep]}\n… [{len(text) - 2 * keep} characters cut] …\n{text[-keep:]}"


def run_steps(argv, env, alive, cwd, limit_s=None, tail=1500):
    """Run one command, writing an alive line every ALIVE_S seconds even while it prints nothing (pip is silent
    while it downloads; dfu-util while it writes); returns (exit code, the output's tail). After `limit_s` seconds
    the command is killed and the exit code is None."""
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd,
                          errors="replace") as p:
        out = []
        reader = threading.Thread(target=lambda: out.extend(p.stdout), daemon=True)
        reader.start()
        start = last = time.monotonic()
        while p.poll() is None:
            time.sleep(0.05)
            if limit_s is not None and time.monotonic() - start > limit_s:
                p.kill()
                p.wait()
                reader.join(5)
                return None, head_tail("".join(out), tail)
            if time.monotonic() - last >= ALIVE_S:
                alive(len(out))
                last = time.monotonic()
        reader.join(5)
        return p.returncode, head_tail("".join(out), tail)


# --- the N6 (S7 bite 4b; D82, D82a, D83): the mpremote and dfu-util command lines RESTORE.md uses ---------------
N6_DFU = "37c5:9206"           # the N6's bootloader: BOOTLOADER(0) FIRMWARE(1) FILESYSTEM(2) ROMFS0(3) (ml/README.md)
ROMFS_ALT = 3                  # the model area: the ONLY alt this agent ever writes (alt 0 is never written)
BOOT_ALT = 2                   # a read of FILESYSTEM with -R boots the firmware (ml/README.md :124)
PARTITION = 0x01800000         # OMV_ROMFS_PART0_LENGTH, 24 MiB (boards/OPENMV_N6/board_config.h)
RESTORE_FAILED = "restore_failed"
TOOL_S = 60.0                  # one mpremote call or dfu-util listing; longer is a fault
FLASH_S = 900.0                # one 24 MiB write or read over DFU
BACK_S = 60.0                  # the N6 answering again after a boot, or its bootloader after machine.bootloader()
N6_PORTS = "/dev/serial/by-id/*"
MARK = "NB:"                   # the line a board script prints its JSON on (JSON lines, never base64: hil_board.py)

ROM_SCRIPT = """
import os, json
r = {}
for n in sorted(os.listdir('/rom')):
    r[n] = os.stat('/rom/' + n)[6]
print('NB:' + json.dumps(r))
"""
INFO_SCRIPT = """
import os, sys, gc, json, omv
gc.collect()
print('NB:' + json.dumps({'board_type': omv.board_type(), 'firmware': sys.version, 'machine': os.uname().machine,
                          'mem_free': gc.mem_free()}))
"""
# D83 on a camera: free memory before and after the load, 2 warm-ups, ticks_us around each of N predicts, free memory
# after. D83a (Lead call, lead 2): the input is made on the board from seed 0 (nothing is written to the camera's
# /flash): every byte of the image's buffer from the LCG in `synthetic_bytes`, and the board sends the buffer's sha256,
# which the laptop compares with its own from the same generator.
LCG_A, LCG_C, LCG_M = 1103515245, 12345, 0x7FFFFFFF


def synthetic_bytes(w, h, c, seed=SEED):
    """The N6's input buffer (D83a): grayscale (1 byte a pixel) for one channel, else RGB565 (2 bytes a pixel)."""
    out, x = bytearray(w * h * (1 if c == 1 else 2)), seed
    for i in range(len(out)):
        x = (LCG_A * x + LCG_C) & LCG_M
        out[i] = (x >> 16) & 0xFF
    return bytes(out)


# The N6's timing in steps (Lead call, lead 2, after card 6: the one-exec script reset the board with no traceback).
# Each stage is its own `mpremote exec` and prints its marker, so the run log shows the last stage that answered.
# Stage 1 starts from a soft reset; stages 2-6 use mpremote's `resume` (no soft reset), so the board keeps their state.
# D83: free memory before and after the load; stage 5's predict is the first warm-up, stage 6 does the second and
# then times each of N predicts with ticks_us. D83a: the input buffer from the LCG in `synthetic_bytes`, its sha256 sent.
STAGES = [
    ("import", """
import gc, json, time, ml, image, hashlib, binascii
gc.collect()
nb_before = gc.mem_free()
print('NB:' + json.dumps({'stage': 1, 'mem_before_load': nb_before}))
"""),
    ("ml.Model", """
m = ml.Model(%(path)r)
gc.collect()
nb_loaded = gc.mem_free()
print('NB:' + json.dumps({'stage': 2, 'mem_after_load': nb_loaded}))
"""),
    ("input_shape", """
s = m.input_shape[0]
h, w, c = s[-3], s[-2], s[-1]
print('NB:' + json.dumps({'stage': 3, 'input_shape': list(s)}))
"""),
    ("seeded image", """
img = image.Image(w, h, image.GRAYSCALE if c == 1 else image.RGB565)
b = img.bytearray()
x = %(seed)d
for i in range(len(b)):
    x = (%(a)d * x + %(c)d) & %(m)d
    b[i] = (x >> 16) & 0xff
sha = binascii.hexlify(hashlib.sha256(b).digest()).decode()
print('NB:' + json.dumps({'stage': 4, 'input_sha256': sha, 'input_bytes': len(b)}))
"""),
    ("one predict", """
m.predict([img])
print('NB:' + json.dumps({'stage': 5}))
"""),
    ("N windows", """
m.predict([img])
total = 0
lo = hi = None
for _ in range(%(n)d):
    t = time.ticks_us()
    m.predict([img])
    d = time.ticks_diff(time.ticks_us(), t)
    total += d
    lo = d if lo is None or d < lo else lo
    hi = d if hi is None or d > hi else hi
gc.collect()
print('NB:' + json.dumps({'stage': 6, 'us_total': total, 'us_min': lo, 'us_max': hi, 'mem_after': gc.mem_free()}))
"""),
]
BOARD_RESET, SCRIPT_FAILED = "board_reset", "script_failed"



def board_json(out):
    """The JSON a board script printed after MARK, or None."""
    for line in reversed(out.splitlines()):
        if line.startswith(MARK):
            try:
                return json.loads(line[len(MARK):])
            except ValueError:
                return None
    return None


def sha256_head(path, n):
    """sha256 of the first n bytes of a file (a read-back runs to the partition's end and beyond)."""
    h, left = hashlib.sha256(), n
    with open(path, "rb") as f:
        while left > 0:
            block = f.read(min(1 << 20, left))
            if not block:
                break
            h.update(block)
            left -= len(block)
    return h.hexdigest(), n - left


N6_PORT_NAME = "Pyboard"       # RESTORE.md: the N6's /dev/serial/by-id name holds Pyboard, the AE3's OpenMV_OpenMV_Camera


class N6:
    """The N6 on the Pi's USB. Found by asking it (omv.board_type(), D81), and only ports whose name holds Pyboard
    are ever asked: the AE3's port is never touched on this route (QE PR 123: one operation at a time on the AE3).
    Every dfu-util call names the N6's bootloader by its serial (-S), the one recorded with the backup, and refuses
    when more than one 37c5:9206 device is present (QE PR 123)."""

    def __init__(self, folder, mpremote, dfu_util, ports_glob, alive):
        self.folder, self.mpremote, self.dfu_util, self.ports_glob = folder, mpremote, dfu_util, ports_glob
        self.alive, self.port, self.serial = alive, None, None

    def waiting(self, last):
        """One second of a wait for the bootloader or the boot, with an alive line every ALIVE_S seconds (QE PR 123:
        the laptop's 60 s of silence must not end a wait that is up to 60 s long itself)."""
        time.sleep(1)
        if time.monotonic() - last >= ALIVE_S:
            self.alive(0)
            return time.monotonic()
        return last

    def masked(self, text):
        """No board serial in agent.log or in an answer (QE PR 126 nit 1: the run folder must not carry the unit's
        serial either): ours reads <n6_dfu_serial>, any other only by its last 4 digits."""
        if self.serial:
            text = text.replace(self.serial, "<n6_dfu_serial>")
        return re.sub(r'serial="([^"<]*)"', lambda m: f'serial="…{m.group(1)[-4:]}"', text)

    def run(self, argv, limit_s, tail=1500):
        """One tool call; its command line, exit code and output go to stderr, which is the run's agent.log (QE 10:
        dfu-util's own words must be kept)."""
        rc, out = run_steps(argv, None, self.alive, self.folder, limit_s, tail)
        shown = [a if len(a) < 200 else a[:60] + f"… ({len(a)} chars)" for a in argv]     # scripts are long
        sys.stderr.write(self.masked(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(shown)}\nexit {rc}\n{out.rstrip()}\n"))
        sys.stderr.flush()
        return rc, out

    def exec(self, port, code, limit_s=TOOL_S):
        return self.run([self.mpremote, "connect", port, "exec", code], limit_s)

    def asks_n6(self, port):
        code, out = self.exec(port, "import omv; print('NB:' + omv.board_type())", 15)
        return code == 0 and (MARK + "N6") in out

    def find(self):
        if self.port:                              # once found, only the N6's own port is asked (never the AE3's)
            if os.path.exists(self.port) and self.asks_n6(self.port):
                return self.port
            raise Stop(NO_ANSWER, f"the N6 does not answer on {os.path.basename(self.port)}")
        ports = sorted(p for p in glob.glob(self.ports_glob) if N6_PORT_NAME in os.path.basename(p))
        for p in ports:
            if self.asks_n6(p):
                self.port = p
                return p
        raise Stop(NO_ANSWER, f"no N6 answered on the Pi's USB (asked {len(ports)} {N6_PORT_NAME} port(s): "
                              f"{', '.join(os.path.basename(p) for p in ports) or 'none'}). It may be in its bootloader: "
                              "a restore-only card writes the backup and boots it")

    def script(self, code, limit_s=TOOL_S):
        rc, out = self.exec(self.find(), code, limit_s)
        found = board_json(out)
        if rc is None:
            raise Stop(NO_ANSWER, f"the N6 did not finish its script in {limit_s:g} s: {out.strip()[-400:]}")
        if rc != 0 or found is None:
            raise Stop(LOAD_FAILED, f"the N6's script failed (exit {rc}): {out.strip()[-600:]}")
        return found

    def rom(self):
        return self.script(ROM_SCRIPT)

    def stage(self, k, name, code, limit_s, resume, port):
        """Stage k of the timing (STAGES): its own exec on `port`, found once before stage 1; nothing else is sent to
        the board between stages (QE PR 130: any exec without `resume`, find()'s own question included, soft-resets
        it and wipes what the stages before defined). Its marker must come back. A failure is named by the last
        stage that answered: board_reset when the N6 then shows its bootloader or its port is gone (card 6),
        no_answer on a timeout, else script_failed with the board's own words."""
        argv = [self.mpremote, "connect", port, *(["resume"] if resume else []), "exec", code]
        rc, out = self.run(argv, limit_s, tail=3000)
        found = board_json(out)
        if rc == 0 and found is not None and found.get("stage") == k:
            return {key: v for key, v in found.items() if key != "stage"}
        last = f"stage {k - 1} ({STAGES[k - 2][0]})" if k > 1 else "none"
        where = f"stage {k} ({name}) did not answer; the last stage that answered: {last}"
        extra = {"stage_failed": k, "stage_name": name, "last_stage": k - 1}
        if rc is None:
            raise Stop(NO_ANSWER, f"{where}: no end within {limit_s:g} s. {out.strip()[-400:]}", **extra)
        gone = not os.path.exists(port)
        try:
            in_dfu = self.in_dfu()
        except Stop:
            in_dfu = True                            # a bootloader is there, whoever's: the board is not running
        if gone or in_dfu:
            raise Stop(BOARD_RESET, f"{where}: the N6 reset itself ({'its port dropped' if gone else 'it'}"
                                    f"{' and it is in its bootloader' if in_dfu else ''}). Board output: "
                                    f"{out.strip()[-600:]}", **extra)
        raise Stop(SCRIPT_FAILED, f"{where}: the board's own words: {out.strip()[-800:]}", **extra)

    def dfu(self, *args):
        """A dfu-util command line for the pinned N6 only."""
        if not self.serial:
            raise Stop(LOAD_FAILED, "no DFU serial was given for the N6: nothing is written to an unpinned device")
        return [self.dfu_util, "-d", N6_DFU, "-S", self.serial, *args]

    def in_dfu(self):
        """True when exactly one 37c5:9206 bootloader is present, it is ours (its serial) and alt 3 is ROMFS0."""
        rc, out = self.run([self.dfu_util, "-d", N6_DFU, "-l"], TOOL_S)
        lines = [l for l in out.splitlines() if l.startswith("Found DFU") and N6_DFU in l.lower()]
        if rc != 0 or not lines:
            return False
        field = lambda key, line: (re.search(key + r'="([^"]*)"', line) or re.search(r"(.*)", line)).group(1)
        devices = {field("path", l) for l in lines}             # one line per alt; one path per device
        serials = {field("serial", l) for l in lines}
        if len(devices) != 1 or len(serials) != 1:
            raise Stop(LOAD_FAILED, f"{len(devices)} N6 bootloaders ({N6_DFU}) are on the USB: nothing is written "
                                    "until only the bench N6 is in its bootloader. If ours was sent there, it stays in "
                                    "it: a restore-only card boots it once the other device is gone")
        if serials != {self.serial}:
            raise Stop(LOAD_FAILED, f"the N6 bootloader on the USB has a serial ending {sorted(serials)[0][-4:]}, not "
                                    "[bench] n6_dfu_serial: another board; nothing is written. If ours was sent to its "
                                    "bootloader, it stays there: a restore-only card boots it once the other is gone")
        return any(f'alt={ROMFS_ALT}, name="ROMFS0"' in l for l in lines)

    def enter_dfu(self):
        """The N6's bootloader, with alt 3 named ROMFS0 (checked on the day, as RESTORE.md says)."""
        if self.in_dfu():
            return
        self.exec(self.find(), "import machine; machine.bootloader()", 15)   # an error as the port drops is normal
        deadline, last = time.monotonic() + BACK_S, time.monotonic()
        while time.monotonic() < deadline:
            if self.in_dfu():
                return
            last = self.waiting(last)
        raise Stop(NO_ANSWER, f"the N6's bootloader ({N6_DFU}, alt {ROMFS_ALT} = ROMFS0) did not appear within "
                              f"{BACK_S:g} s of machine.bootloader(). The N6 may be in its bootloader anyway: a "
                              "restore-only card writes the backup and boots it")

    def write(self, path, alt=ROMFS_ALT):
        if alt != ROMFS_ALT:                       # D82: alt 0 (and every other alt) is never written
            raise Stop(LOAD_FAILED, f"refused: only alt {ROMFS_ALT} (ROMFS0) is ever written, not alt {alt}")
        rc, out = self.run(self.dfu("-a", str(ROMFS_ALT), "-D", path), FLASH_S)
        if rc != 0:
            raise Stop(LOAD_FAILED, f"dfu-util could not write ROMFS0 (exit {rc}): {out.strip()[-400:]}")

    def read_back(self, n):
        """sha256 of the first n bytes of ROMFS0 read back over DFU (verify by read-back, not by 'Download done')."""
        check = os.path.join(self.folder, "models", "n6_readback.bin")
        if os.path.exists(check):
            os.remove(check)                       # dfu-util does not write over a file
        rc, out = self.run(self.dfu("-a", str(ROMFS_ALT), "-U", check), FLASH_S)
        try:
            if rc != 0 or not os.path.isfile(check):
                raise Stop(READBACK, f"dfu-util could not read ROMFS0 back (exit {rc}): {out.strip()[-400:]}")
            have, got = sha256_head(check, n)
            if got < n:
                raise Stop(READBACK, f"the read-back holds {got:,} bytes, fewer than the {n:,} written")
            return have
        finally:
            if os.path.exists(check):
                os.remove(check)

    def boot(self):
        """Leave the bootloader (a read of alt 2 with -R; exit 251 is normal), then wait for the N6 to answer."""
        junk = os.path.join(self.folder, "models", "n6_boot.bin")
        if os.path.exists(junk):
            os.remove(junk)
        self.run(self.dfu("-a", str(BOOT_ALT), "-U", junk, "-R"), FLASH_S)
        if os.path.exists(junk):
            os.remove(junk)
        deadline, last = time.monotonic() + BACK_S, time.monotonic()
        while time.monotonic() < deadline:
            try:
                return self.find()
            except Stop:
                last = self.waiting(last)
        raise Stop(NO_ANSWER, f"the N6 did not answer within {BACK_S:g} s of leaving its bootloader")

    def flash(self, path, length, want, rom):
        """Write `path` to ROMFS0, read it back (compare capped at `length`), boot, and check /rom against `rom`."""
        self.enter_dfu()
        self.write(path)
        have = self.read_back(length)
        if have != want:
            raise Stop(READBACK, f"ROMFS0 read back differs: sha256 {have[:16]}… of the first {length:,} bytes, "
                                 f"written {want[:16]}…")
        self.boot()
        listing = self.rom()
        if listing != rom:
            missing = sorted(set(rom) - set(listing))
            extra = sorted(set(listing) - set(rom))
            sizes = sorted(k for k in set(rom) & set(listing) if rom[k] != listing[k])
            raise Stop(LOAD_FAILED, f"/rom does not list the image: missing {missing}, extra {extra}, "
                                    f"other size {sizes}")
        return have, listing


def check_memory(what):
    """The D89 floor: refuse `what` when free memory is known and under 50 MB (unknown on the laptop: no check)."""
    free = proc_kib("/proc/meminfo", "MemAvailable")
    if free is not None and free < MEM_FLOOR:
        raise Stop(LOW_MEMORY, f"only {free / 1048576:.1f} MB free on the Pi (MemAvailable); {what} needs at least "
                               f"{MEM_FLOOR // 1048576} MB free (D89)")
    return free


class Agent:
    def __init__(self, folder, runtime="litert", mpremote="mpremote", dfu_util="dfu-util", ports_glob=N6_PORTS):
        self.folder = os.path.abspath(folder)
        self.models_dir = os.path.join(self.folder, "models")
        self.runtime = RUNTIMES[runtime]()
        self.loaded = {}          # name -> (runtime model, threads)
        self.mem_before_load = self.mem_after_load = None
        self.alive = lambda done: None
        self.n6 = N6(self.folder, mpremote, dfu_util, ports_glob, lambda done: self.alive(done))

    def handle(self, req):
        op = req.get("op")
        kind = req.get("kind", "pi")
        if kind == "n6" and op in ("info", "load", "time", "reset", "restore", "hard_reset"):
            if req.get("dfu_serial"):
                self.n6.serial = str(req["dfu_serial"])
            return getattr(self, "n6_" + op)(req)
        fn = getattr(self, "op_" + str(op), None)
        if fn is None:
            raise Stop(LOAD_FAILED, f"unknown op {op!r}")
        if op in ("info", "load", "time", "reset") and kind != "pi":
            raise Stop(NOT_BUILT, f"the {kind} route is not built yet (S7 bite 5); this agent runs the Pi and the N6")
        return fn(req)

    # --- the N6 (S7 bite 4b) ---
    def _n6_file(self, name, want):
        path = os.path.join(self.models_dir, file_name(name))
        if not os.path.isfile(path):
            raise Stop(LOAD_FAILED, f"{name} is not in {self.models_dir}: the copy did not arrive", written=False)
        have = sha256(path)
        if have != want:
            raise Stop(READBACK, f"{name} on the Pi has sha256 {have[:16]}…, the laptop sent {want[:16]}…",
                       written=False)
        return path

    def op_file_sha256(self, req):
        """sha256 of a file in models/, or None: the laptop copies the N6 backup only when it is not there already."""
        path = os.path.join(self.models_dir, file_name(req["name"]))
        return {"sha256": sha256(path) if os.path.isfile(path) else None}

    def n6_info(self, req):
        if self.n6.in_dfu():                       # left in its bootloader (a Cancel, a dropped link): a normal start
            return {"kind": "n6", "state": "bootloader", "board_type": "N6", "firmware": None, "mem_free": None,
                    "rom": None, "model_area_bytes": PARTITION}
        found = self.n6.script(INFO_SCRIPT)
        return {"kind": "n6", **found, "port": os.path.basename(self.n6.port or ""), "rom": self.n6.rom(),
                "model_area_bytes": PARTITION}

    def n6_load(self, req):
        """Write our image (the backup's entries + ours, D82a) to ROMFS0 and check it. Both files are checked
        before anything is written; a Stop says whether the write had started ("written"), so the laptop knows
        whether the restore must run."""
        image = self._n6_file(req["image"], req["image_sha256"])
        self._n6_file(req["backup"], req["backup_sha256"])       # the way back is on the Pi before the first write
        if os.path.getsize(image) != int(req["image_len"]) or int(req["image_len"]) > PARTITION:
            raise Stop(LOAD_FAILED, f"the image is {os.path.getsize(image):,} bytes, the laptop said "
                                    f"{req['image_len']:,} (model area {PARTITION:,})", written=False)
        try:
            self.n6.enter_dfu()
        except Stop as e:
            e.extra["written"] = False
            raise
        try:
            have, listing = self.n6.flash(image, int(req["image_len"]), req["image_sha256"], req["rom"])
        except Stop as e:
            e.extra["written"] = True
            raise
        return {"loaded": sorted(req.get("ours", [])), "route": f"ROMFS0, DFU alt {ROMFS_ALT}", "readback_sha256": have,
                "rom": listing, "port": os.path.basename(self.n6.port or "")}

    def n6_restore(self, req):
        """The backup's partition back into ROMFS0, read back whole, booted, /rom equal to the backup's listing."""
        try:
            backup = self._n6_file(req["backup"], req["backup_sha256"])
            if os.path.getsize(backup) != PARTITION:
                raise Stop(RESTORE_FAILED, f"the backup on the Pi is {os.path.getsize(backup):,} bytes, not the "
                                           f"partition's {PARTITION:,}")
            have, listing = self.n6.flash(backup, PARTITION, req["backup_sha256"], req["rom"])
        except Stop as e:
            raise Stop(RESTORE_FAILED, f"the restore of ROMFS0 did not finish: {e}") from None
        return {"restored": True, "readback_sha256": have, "rom": listing}

    def n6_time(self, req):
        """One window in STAGES; a failure says which stage was the last to answer (and its stop rule)."""
        n = int(req["n"])
        if n < 1:
            raise Stop(LOAD_FAILED, "a window needs at least one inference")
        fill = {"path": "/rom/" + req["rom_name"], "n": n, "seed": SEED, "a": LCG_A, "c": LCG_C, "m": LCG_M}
        got, port = {}, self.n6.find()               # asked once, before stage 1 (which starts fresh anyway)
        for k, (name, code) in enumerate(STAGES, 1):
            limit = TOOL_S + (2.0 * (n + 1) if k == 6 else 0)
            got.update(self.n6.stage(k, name, code % fill, limit, resume=k > 1, port=port))
        return {"n": n, "warmups": WARMUPS, "ms_mean": got["us_total"] / n / 1000, "ms_min": got["us_min"] / 1000,
                "ms_max": got["us_max"] / 1000, "mem_before_load": got["mem_before_load"],
                "mem_after_load": got["mem_after_load"], "mem_after": got["mem_after"], "input_shape": got["input_shape"],
                "input_sha256": got.get("input_sha256"), "input_bytes": got.get("input_bytes"),
                "stages": [name for name, _ in STAGES]}

    def n6_reset(self, req):
        rc, out = self.n6.run([self.n6.mpremote, "connect", self.n6.find(), "soft-reset"], TOOL_S)
        if rc != 0:
            raise Stop(NO_ANSWER, f"the N6 did not take a soft reset (exit {rc}): {out.strip()[-300:]}")
        return {"reset": True}

    def n6_hard_reset(self, req):
        """The card's last step (lead 2, after PR 123): `mpremote reset`, so the N6's own main.py runs again and the
        board is left as found. Nothing is asked of it afterwards (that would stop main.py again). A board in its
        bootloader is booted first (a read of alt 2 with -R, no write) when the laptop says our image is not on it
        (QE PR 130: a /rom card's board that reset itself); otherwise it is refused (a restore must come first)."""
        if self.n6.in_dfu():
            if not req.get("boot_if_in_bootloader"):
                raise Stop(NO_ANSWER, "the N6 is in its bootloader, not booted: run a restore-only card, which boots it")
            self.n6.boot()                           # no write: the model area is the backup's (nothing of ours there)
        rc, out = self.n6.run([self.n6.mpremote, "connect", self.n6.find(), "reset"], TOOL_S)
        if rc != 0:
            raise Stop(NO_ANSWER, f"the N6 did not take a hard reset (exit {rc}): {out.strip()[-300:]}")
        return {"hard_reset": True, "port": os.path.basename(self.n6.port or "")}

    def op_hello(self, req):
        return {"agent": VERSION, "python": platform.python_version(), "machine": platform.machine(),
                "system": platform.platform(), "folder": self.folder, "runtime": self.runtime.name}

    def op_info(self, req):
        version = getattr(self.runtime, "version", lambda: self.runtime.name)()
        return {"kind": "pi", "firmware": platform.platform(), "runtime": version,
                "mem_free": proc_kib("/proc/meminfo", "MemAvailable"), "model_area_bytes": 0}

    def op_load(self, req):
        threads = int(req.get("threads", 1))
        self.loaded = {}
        self.mem_before_load = check_memory("a load")
        for name, want in sorted(req["files"].items()):
            path = os.path.join(self.models_dir, file_name(name))
            if not os.path.isfile(path):
                raise Stop(LOAD_FAILED, f"{name} is not in {self.models_dir}: the copy did not arrive")
            have = sha256(path)
            if have != want:
                raise Stop(READBACK, f"{name} on the Pi has sha256 {have[:16]}…, the laptop sent {want[:16]}…")
            try:
                self.loaded[name] = (self.runtime.open(path, threads), threads)
            except Exception as e:                     # the runtime's own words go back to the laptop
                raise Stop(LOAD_FAILED, f"{self.runtime.name} could not load {name}: {type(e).__name__}: {e}")
        self.mem_after_load = proc_kib("/proc/meminfo", "MemAvailable")
        return {"loaded": sorted(self.loaded), "route": "agent folder", "threads": threads,
                "io": {n: self.runtime.describe(m) for n, (m, _) in self.loaded.items()},
                "mem_free": proc_kib("/proc/meminfo", "MemAvailable"), "vm_hwm": proc_kib("/proc/self/status", "VmHWM")}

    def op_time(self, req):
        name, n = req["name"], int(req["n"])
        if name not in self.loaded:
            raise Stop(NOT_LOADED, f"{name} is not loaded on the Pi")
        if n < 1:
            raise Stop(LOAD_FAILED, "a window needs at least one inference")
        model, threads = self.loaded[name]
        check_memory("a window")
        for _ in range(WARMUPS):
            self.runtime.invoke(model)
        each, last = [], time.monotonic()
        for _ in range(n):
            t = time.perf_counter_ns()
            self.runtime.invoke(model)
            each.append((time.perf_counter_ns() - t) / 1e6)
            if len(each) < n and time.monotonic() - last >= ALIVE_S:     # none after the last: the answer follows
                self.alive(len(each))
                last = time.monotonic()
        return {"n": n, "warmups": WARMUPS, "threads": threads, "ms_mean": sum(each) / n, "ms_min": min(each),
                "ms_max": max(each), "mem_before_load": self.mem_before_load, "mem_after_load": self.mem_after_load,
                "mem_after": proc_kib("/proc/meminfo", "MemAvailable"), "vm_hwm": proc_kib("/proc/self/status", "VmHWM")}

    def op_reset(self, req):
        self.loaded = {}
        return {"reset": True}

    def _install(self, req):
        name = req.get("runtime", "litert")
        if name not in INSTALLS:
            raise Stop(INSTALL_FAILED, f"no install is defined for runtime {name!r}")
        spec = INSTALLS[name]
        venv = os.path.join(self.folder, spec["venv"])
        req_file = os.path.join(self.folder, spec["requirements"])
        if not os.path.isfile(req_file):
            raise Stop(INSTALL_FAILED, f"{spec['requirements']} is not in {self.folder}: copy the agent again")
        return spec, venv, req_file, os.path.join(venv, "installed.json")

    def op_install_status(self, req):
        spec, venv, req_file, record = self._install(req)
        try:
            with open(record) as f:
                done = json.load(f)
        except (OSError, ValueError):
            return {"ready": False, "venv": spec["venv"]}
        return {"ready": done.get("requirements_sha256") == sha256(req_file), "venv": spec["venv"], "installed": done}

    def op_install(self, req):
        """Build the runtime's environment (D89). Every write stays inside the agent's folder: the venv, pip's
        temporary files (TMPDIR) and no cache; pip reads no config file (the Pi's points at piwheels too)."""
        spec, venv, req_file, record = self._install(req)
        check_memory("an install")                    # pip on a 512 MB Pi (QE PR 113 nit 4)
        if os.path.isdir(venv):
            shutil.rmtree(venv)                       # a half-made environment from an earlier try
        tmp = os.path.join(self.folder, "tmp")
        os.makedirs(tmp, exist_ok=True)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": self.folder, "TMPDIR": tmp,
               "PIP_CONFIG_FILE": os.devnull, "PIP_NO_CACHE_DIR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
               "PYTHONNOUSERSITE": "1"}
        py = os.path.join(venv, "bin", "python")
        steps = [("make the environment", [sys.executable, "-m", "venv", venv]),
                 ("install the pinned packages", [py, "-m", "pip", *PIP_ARGS, req_file])]
        for what, argv in steps:
            code, out = (fake_step(argv, venv) if FAKE_INSTALL else run_steps(argv, env, self.alive, self.folder))
            if code != 0:
                shutil.rmtree(venv, ignore_errors=True)
                hint = (" python3's venv module does not work here; D89: no apt, so the install stops"
                        if what == "make the environment" else "")
                raise Stop(INSTALL_FAILED, f"could not {what} (exit code {code}).{hint}\n{out}")
        listing = ("import json, importlib.metadata as m; print(json.dumps({d.metadata['Name']: [d.version, "
                   "d.metadata.get('License-Expression') or d.metadata.get('License') or ''] for d in m.distributions()}))")
        code, out = fake_step([py, "-c", listing], venv) if FAKE_INSTALL else run_steps([py, "-c", listing], env, self.alive, self.folder)
        packages = json.loads(out.strip().splitlines()[-1]) if code == 0 else {}
        done = {"runtime": req.get("runtime", "litert"), "venv": spec["venv"], "python": os.path.join(spec["venv"], "bin", "python"),
                "requirements_sha256": sha256(req_file), "packages": packages,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(record, "w") as f:
            json.dump(done, f, indent=1)
        shutil.rmtree(tmp, ignore_errors=True)
        return {"built": True, "installed": done}

    def op_bye(self, req):
        return {"bye": True}


def fake_step(argv, venv):
    """--fake-install: the venv is a folder whose python is this one; pip and the listing are pretended."""
    if argv[1:3] == ["-m", "venv"]:
        os.makedirs(os.path.join(venv, "bin"), exist_ok=True)
        os.symlink(sys.executable, os.path.join(venv, "bin", "python"))
        return 0, ""
    if "-c" in argv:
        return 0, json.dumps({"ai-edge-litert": ["2.2.0", "Apache 2.0"]})
    return 0, "pretended: " + " ".join(argv[1:])


def serve(agent, lines, write):
    """Answer each request line; a request that cannot be read is answered too, never silently dropped."""
    for line in lines:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("not an object")
            rid = req.get("id")
        except ValueError:
            write({"id": None, "ok": False, "stop": LOAD_FAILED, "error": "the request is not a JSON object"})
            continue
        agent.alive = lambda done, rid=rid: write({"id": rid, "alive": True, "done": done})
        try:
            write({"id": rid, "ok": True, **agent.handle(req)})
        except Stop as e:
            write({**e.extra, "id": rid, "ok": False, "stop": e.stop, "error": agent.n6.masked(str(e))})
        except Exception as e:                         # a fault in the agent itself: reported, the agent keeps going
            write({"id": rid, "ok": False, "stop": LOAD_FAILED, "error": f"agent fault: {type(e).__name__}: {e}"})
        if req.get("op") == "bye":
            return


def main(argv):
    global ALIVE_S
    folder = argv[1]
    opt = lambda name, default: argv[argv.index(name) + 1] if name in argv else default
    runtime = opt("--runtime", "litert")
    ALIVE_S = float(opt("--alive-s", ALIVE_S))
    FakeRuntime.ms = float(opt("--fake-ms", 0))          # tests only: how long one fake inference takes
    global FAKE_INSTALL
    FAKE_INSTALL = "--fake-install" in argv
    os.makedirs(os.path.join(folder, "models"), exist_ok=True)
    mpremote = os.path.expanduser("~/mpv/bin/mpremote")             # the Pi's mpremote (RESTORE.md)
    tools = {"mpremote": opt("--mpremote", mpremote if os.path.exists(mpremote) else "mpremote"),
             "dfu_util": opt("--dfu-util", "dfu-util"), "ports_glob": opt("--ports", N6_PORTS)}
    global PARTITION, BACK_S
    PARTITION = int(opt("--partition", PARTITION))                # tests only: a small fake model area
    BACK_S = float(opt("--back-s", BACK_S))

    def write(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    serve(Agent(folder, runtime, **tools), sys.stdin, write)


if __name__ == "__main__":
    main(sys.argv)
