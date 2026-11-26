"""S5 bite 3a: the label Review page and its routes, on synthetic frames labeled by the fake SAM
(tests/fake_sam.py): page, frame view, picture modes, click-to-select, actions with refusals."""

import io

from PIL import Image

from tests.fake_sam import run_sam
from tests.test_preview import make_project

F = "v_aaaaaa_0.png"   # synthetic_frame: urchins at (960-1060, 300-400) and (200-260, 200-260),
                       # a sea star at (2000-2100, 500-600), a kelp band y 1000-1200


def labeled(tmp_path, n=2):
    client, folder = make_project(tmp_path, n)
    out = run_sam(folder, [f"v_aaaaaa_{i}.png" for i in range(n)])
    assert out["done"], out["log"]
    return client, folder


def test_page_lists_labeled_frames_and_the_project_links_it(tmp_path):
    client, folder = labeled(tmp_path)
    page = client.get("/p/p/labels").text
    assert "Review labels" in page and F in page and "v_aaaaaa_1.png" in page
    assert 'id="todo-count">2<' in page and "labelreview.js" in page and "viewer.js" in page
    assert 'href="/p/p/labels"' in client.get("/p/p").text


def test_page_without_results_says_run_sam_first(tmp_path):
    client, folder = make_project(tmp_path, 1)
    assert "run SAM on the" in client.get("/p/p/labels").text


def test_frame_view_click_select_tag_accept_and_undo(tmp_path):
    client, folder = labeled(tmp_path)
    v = client.get(f"/p/p/labels/frame/{F}").json()
    assert (v["width"], v["height"]) == (2600, 1200) and len(v["untagged"]) == 2
    kinds = sorted((m["kind"], m["name"]) for m in v["masks"])
    assert ("area", "kelp") in kinds and ("object", "sea star") in kinds
    # click inside the far urchin selects it; on the kelp band selects the area; on water: none
    urchin = client.get(f"/p/p/labels/at/{F}?x=1010&y=350").json()["mask_id"]
    assert urchin in v["untagged"]
    kelp = client.get(f"/p/p/labels/at/{F}?x=100&y=1100").json()["mask_id"]
    assert next(m for m in v["masks"] if m["id"] == kelp)["kind"] == "area"
    assert client.get(f"/p/p/labels/at/{F}?x=1500&y=100").json()["mask_id"] is None
    # accept is refused while urchins are untagged; nothing changes
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "accept"})
    assert r.status_code == 400 and "no type yet" in r.json()["error"]
    # a urchin cannot become kelp (an area)
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "class", "mask_ids": [urchin],
                                                   "class_id": 9})
    assert r.status_code == 400 and "same kind" in r.json()["error"]
    for mid in v["untagged"]:
        r = client.post(f"/p/p/labels/act/{F}", json={"action": "class", "mask_ids": [mid],
                                                       "class_id": 1})
        assert r.status_code == 200 and r.json()["note"] == "class changed"
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "accept"})
    assert r.status_code == 200 and r.json()["state"]["accepted"]
    assert 'id="todo-count">1<' in client.get("/p/p/labels").text
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "undo"})
    assert r.json()["note"] == "undone: accept" and not r.json()["state"]["accepted"]


def test_delete_hides_the_mask_and_undo_brings_it_back(tmp_path):
    client, folder = labeled(tmp_path)
    star = client.get(f"/p/p/labels/at/{F}?x=2050&y=550").json()["mask_id"]
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "delete", "mask_ids": [star]})
    assert star not in [m["id"] for m in r.json()["masks"]]
    assert client.get(f"/p/p/labels/at/{F}?x=2050&y=550").json()["mask_id"] is None
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "undo"})
    assert star in [m["id"] for m in r.json()["masks"]]
    assert client.post(f"/p/p/labels/act/{F}", json={"action": "fly"}).status_code == 400
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "delete", "mask_ids": "x"})
    assert r.status_code == 400 and "must be a list of numbers" in r.json()["error"]
    assert client.post(f"/p/p/labels/act/{F}", content=b"{oops",
                       headers={"content-type": "application/json"}).status_code == 400


def test_picture_modes_and_selection_outline(tmp_path):
    client, folder = labeled(tmp_path)
    star = client.get(f"/p/p/labels/at/{F}?x=2050&y=550").json()["mask_id"]
    pics = {}
    for mode in ("all", "objects", "areas", "map"):
        r = client.get(f"/p/p/labels/picture/{F}?mode={mode}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        pics[mode] = Image.open(io.BytesIO(r.content)).convert("RGB")
        assert pics[mode].size == (2600, 1200)               # never scaled up
    px = lambda im, x, y: im.getpixel((x, y))
    assert px(pics["objects"], 100, 1100) != px(pics["areas"], 100, 1100)   # kelp: areas only
    sel = Image.open(io.BytesIO(client.get(f"/p/p/labels/picture/{F}?sel={star}").content))
    r, g, b = sel.convert("RGB").getpixel((2050, 500))       # the star's top edge: white
    assert min(r, g, b) > 200
    assert client.get("/p/p/labels/picture/nope.png").status_code == 404
    assert client.get("/p/p/labels/frame/../x.png").status_code == 404


def test_split_check_fully_checked_and_running_totals(tmp_path):
    client, folder = labeled(tmp_path)
    v = client.get(f"/p/p/labels/frame/{F}").json()
    assert v["state"] == {"accepted": False, "split_ok": None, "checked": []}
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "split", "ok": False})
    assert r.json()["state"]["split_ok"] is False and r.json()["note"] == "split check no"
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "checked", "class_id": 3, "checked": True})
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "checked", "class_id": 1, "checked": True})
    assert r.json()["state"]["checked"] == [1, 3]
    assert client.post(f"/p/p/labels/act/{F}", json={"action": "split", "ok": "maybe"}
                       ).status_code == 400
    # totals count accepted frames only
    page = client.get("/p/p/labels").text
    assert "Running totals" in page and 'max="150" value="0"' in page
    for mid in v["untagged"]:
        client.post(f"/p/p/labels/act/{F}", json={"action": "class", "mask_ids": [mid], "class_id": 1})
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "accept"})
    t = {x["name"]: x for x in r.json()["totals"]}
    assert (t["purple urchin"]["objects"], t["purple urchin"]["frames"]) == (0, 0)   # split "no"
    assert (t["purple urchin"]["split_no"], t["sea star"]["split_no"]) == (2, 1)
    assert "kelp" not in t                                    # areas are not counted objects
    # a change on an accepted frame updates the totals (the cache follows the review)
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "class", "mask_ids": [v["untagged"][0]],
                                                   "class_id": 2})
    t = {x["name"]: x for x in r.json()["totals"]}
    assert (t["purple urchin"]["split_no"], t["red urchin"]["split_no"]) == (1, 1)
    assert r.json()["state"]["accepted"]
    # the frame's split check is "no" (set above): the detector totals leave it out (QE PR 51)
    assert t["purple urchin"]["split_no"] == 1 and t["sea star"]["split_no"] == 1
    r = client.post(f"/p/p/labels/act/{F}", json={"action": "split", "ok": True})
    t = {x["name"]: x for x in r.json()["totals"]}
    assert (t["sea star"]["objects"], t["sea star"]["split_no"]) == (1, 0)
