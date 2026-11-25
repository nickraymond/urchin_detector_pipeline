"""S3 bite 5: the large viewer and class legend on the Labeling page (SPEC Q16), and the
fast overlay (masks decoded inside their box only, cached per file version). Synthetic only."""

import io
import json
import os
import re

import numpy as np
import pytest
from PIL import Image

from app.sam import classes, results
from app.stages import label
from tests.test_clips import wait_for
from tests.test_preview import make_project
from tests.test_sam_runner import FAKE


def rle_of(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(mask.shape), "counts": r["counts"].decode("ascii")}


def box_of(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


@pytest.mark.parametrize("seed", range(12))
def test_decode_box_matches_pycocotools(seed):
    rng = np.random.default_rng(seed)
    h, w = rng.integers(5, 300, 2)
    m = np.zeros((h, w), bool)
    x0, y0 = rng.integers(0, w), rng.integers(0, h)
    x1, y1 = rng.integers(x0 + 1, w + 1), rng.integers(y0 + 1, h + 1)
    m[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0)) > rng.random()
    m[y0, x0] = True
    rle, box = rle_of(m), box_of(m)
    full = results.decode(rle)
    assert np.array_equal(results.decode_box(rle, box), full[box[1]:box[3], box[0]:box[2]])
    assert np.array_equal(results.decode_box(rle, [0, 0, w, h]), full)  # large: C decode


@pytest.mark.parametrize("where", ["corner", "last", "column", "long"])
def test_decode_box_edges(where):
    h, w = 64, 20000
    m = np.zeros((h, w), bool)
    if where == "corner":
        m[0, 0] = True                       # the first run of zeros is empty
    elif where == "last":
        m[-1, -1] = True
    elif where == "column":
        m[:, 1234] = True
    else:
        m[:, 100:2900] = True                # one run of 64 x 2400 > 2^15: several characters
        m[10:20, 2500:2600] = False
    rle, box = rle_of(m), box_of(m)
    got = results.decode_box(rle, box)
    assert np.array_equal(got, m[box[1]:box[3], box[0]:box[2]])
    small = [box[0], box[1], min(box[2], box[0] + 7), box[3]]   # a box cutting a run
    assert np.array_equal(results.decode_box(rle, small), m[small[1]:small[3], small[0]:small[2]])


def labelled(tmp_path):
    client, folder = make_project(tmp_path, n=1)
    label.run(folder, ["v_aaaaaa_0.png"], lambda p, m: None, predictor=FAKE)
    return client, folder, classes.run_units(classes.ensure(folder)["classes"])


def test_overlay_width_and_cache_follow_the_file(tmp_path):
    _, folder, units = labelled(tmp_path)

    def draw(**kw):
        return Image.open(io.BytesIO(label.overlay(folder, "v_aaaaaa_0.png", units, set(), **kw)))
    assert draw().width == label.PREVIEW_WIDTH
    assert draw(width=label.LARGE_WIDTH).width == 2600       # never scaled up
    before = draw()
    p = results.raw_path(folder, "v_aaaaaa_0.png")
    raw = json.loads(p.read_text())
    for u in raw["units"].values():
        u["masks"] = []
    p.write_text(json.dumps(raw))
    os.utime(p, ns=(1, 1))                   # even an older time is a new version
    after = draw()
    at = (round(230 * 1600 / 2600),) * 2     # the urchin's mask is gone after the rewrite
    assert before.getpixel(at) != after.getpixel(at)


def test_overlay_toggles_and_map_through_the_route(tmp_path):
    client, folder, _ = labelled(tmp_path)
    url = "/p/p/labeling/overlay/v_aaaaaa_0.png"
    large = client.get(url, params={"size": "large"})
    assert Image.open(io.BytesIO(large.content)).width == 2600
    assert Image.open(io.BytesIO(client.get(url).content)).width == 1600
    shown = client.get(url, params={"size": "large"}).content
    only = client.get(url, params=[("size", "large")] + [("hide", u) for u in
                      ("sea star", "fish", "kelp", "rock/substrate")]).content
    assert shown != only
    assert client.get(url, params={"map": "1", "size": "large"}).status_code == 200


def test_page_gives_the_viewer_its_frames_and_legend(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path)
    client.post("/p/p/labeling/preview", data={})
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "done", 60)
    page = client.get("/p/p/labeling").text
    assert 'id="viewer"' in page and "viewer.js" in page and 'class="legend v-legend"' in page
    assert 'target="_blank"' not in page                    # never a new tab (SPEC Q16)
    assert "drawn with a box" in page and "areas (kelp, rock) have none" in page
    data = json.loads(re.search(r'<script type="application/json" id="frame-list">(.*?)</script>',
                                page, re.S).group(1))
    assert data["base"] == "/p/p/labeling/overlay/"
    assert [f["file"] for f in data["frames"]] == ["v_aaaaaa_0.png", "v_aaaaaa_2.png",
                                                    "v_aaaaaa_3.png"]
    urchin = next(u for u in data["units"] if u["key"] == "urchin")
    assert urchin["kind"] == "object" and urchin["untagged"] and urchin["colour"].startswith("#")
    assert data["frames"][0]["counts"]["urchin"] == 2
