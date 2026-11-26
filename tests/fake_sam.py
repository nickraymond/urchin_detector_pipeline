"""A stand-in for SAM in tests (no torch, weights or real frames): each prompt finds the
connected blobs of one pure colour in a synthetic image."""

import numpy as np
from scipy import ndimage

COLOURS = {"urchin": (200, 30, 200), "sea star": (250, 140, 0), "kelp": (20, 160, 40),
           "rock": (120, 120, 120)}


class FakePredictor:
    def __init__(self, job):
        self.calls = 0

    def set_image(self, image):
        self.calls += 1
        return np.asarray(image.convert("RGB")).astype(int)

    def find(self, state, prompt):
        colour = COLOURS.get(prompt)
        if colour is None:
            return []
        hit = (np.abs(state - colour).sum(axis=2) < 30)
        labels, n = ndimage.label(hit)
        return [(labels == i, 0.9 - i / 100) for i in range(1, n + 1)]


class FakeClickPredictor:
    """Add a missed object (S5 bite 5): the blob of one pure colour under the click, or the
    largest such blob inside the box; None on water."""
    def __init__(self, job):
        pass

    def click(self, crop, point=None, box=None):
        a = np.asarray(crop.convert("RGB")).astype(int)
        best = None
        for colour in COLOURS.values():
            labels, n = ndimage.label(np.abs(a - colour).sum(axis=2) < 30)
            for i in range(1, n + 1):
                m = labels == i
                if point is not None and m[point[1], point[0]]:
                    return m, 0.95
                if box is not None:
                    x0, y0, x1, y1 = box
                    inside = m[y0:y1, x0:x1].sum()
                    if inside == m.sum() and (best is None or inside > best.sum()):
                        best = m
        return (best, 0.9) if best is not None else None


class SlowFakePredictor(FakePredictor):
    """The fake, slowed down so a test can pause a job mid-run."""
    DELAY = 0.1

    def set_image(self, image):
        import time
        time.sleep(self.DELAY)
        return super().set_image(image)


def run_sam(folder, files, preview=False, predictor="tests.fake_sam:FakePredictor"):
    """Run the SAM worker process once, outside the job module (tests of the worker itself).
    stdin stays open while it runs, as under app/jobctl: end of file would mean "pause".
    Returns {"frames": [item events], "done": the done event or None, "code", "log"}."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    from app.stages import label
    folder = Path(folder)
    job = label.build_job({"folder": str(folder), "files": list(files), "preview": preview,
                           "predictor": predictor})
    path, log = folder / "test_job.json", folder / "test_job.log"
    path.write_text(json.dumps(job))
    with open(log, "w") as err:
        p = subprocess.Popen([sys.executable, "-m", "app.sam.worker", str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
                             cwd=Path(__file__).resolve().parents[1])
        events = [json.loads(l[3:]) for l in p.stdout if l.startswith("@@ ")]
        p.stdin.close()
        code = p.wait()
    return {"frames": [e for e in events if e["event"] == "item"],
            "done": next((e for e in events if e["event"] == "done"), None),
            "code": code, "log": log.read_text()}
