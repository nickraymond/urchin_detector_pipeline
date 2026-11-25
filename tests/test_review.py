"""S2 bite 4: review frames: D (to rejected/), K (kept_hand, D20), Z undo, the page and its
routes. Frames are tiny made-up files or synthetic video; no real footage."""

import sqlite3

import pytest
from PIL import Image

from app import db
from app.stages import extract, review
from tests.test_app_skeleton import BASE, client_for, write_config
from tests.test_clips import wait_for
from tests.test_filters import _project, _where, frame
from tests.test_ingest import digest, make_video


def test_d_k_z_move_files_and_undo_in_order(tmp_path):
    rows = [frame("a_aaaaaa_0.jpg", t=0), frame("a_aaaaaa_1.jpg", t=1, status="dropped_blur")]
    proj, conn = _project(tmp_path, rows)
    assert review.act(conn, proj, "a_aaaaaa_0.jpg", "D")["status"] == "rejected"
    assert review.act(conn, proj, "a_aaaaaa_1.jpg", "K")["status"] == "kept_hand"
    assert _where(proj) == {"a_aaaaaa_0.jpg": "rejected", "a_aaaaaa_1.jpg": "frames"}
    # no-ops add no undo step
    assert review.act(conn, proj, "a_aaaaaa_0.jpg", "D")["changed"] is False
    assert review.act(conn, proj, "a_aaaaaa_1.jpg", "K")["changed"] is False
    assert review.undo(conn, proj) == {"file": "a_aaaaaa_1.jpg", "status": "dropped_blur",
                                       "label": "dropped: blurry", "changed": True}
    assert review.undo(conn, proj)["status"] == "kept"
    assert review.undo(conn, proj) == {}
    assert _where(proj) == {"a_aaaaaa_0.jpg": "frames", "a_aaaaaa_1.jpg": "dropped/blurry"}
    # K on a rejected frame brings it back, kept by hand
    review.act(conn, proj, "a_aaaaaa_0.jpg", "D")
    assert review.act(conn, proj, "a_aaaaaa_0.jpg", "K")["status"] == "kept_hand"
    with pytest.raises(review.ReviewError):
        review.act(conn, proj, "a_aaaaaa_0.jpg", "X")
    with pytest.raises(review.ReviewError):
        review.act(conn, proj, "nope.jpg", "D")


def test_undo_skips_and_keeps_a_step_whose_frame_changed_since(tmp_path):
    proj, conn = _project(tmp_path, [frame("a_aaaaaa_0.jpg"),
                                     frame("a_aaaaaa_1.jpg", t=1, status="dropped_blur")])
    review.act(conn, proj, "a_aaaaaa_0.jpg", "D")
    review.act(conn, proj, "a_aaaaaa_1.jpg", "K")            # kept by hand
    # a new plan sets the kept-by-hand frame aside (D17a, Lead call on PR 14)
    conn.execute("UPDATE frames SET status='unplanned' WHERE file='a_aaaaaa_1.jpg'")
    extract.move_home(proj, "a_aaaaaa_1.jpg", "unplanned")
    assert review.undo(conn, proj)["file"] == "a_aaaaaa_0.jpg"  # the older, still-valid step
    with pytest.raises(review.ReviewError, match="changed"):
        review.undo(conn, proj)
    # the K step is kept, so the frame comes back kept by hand (QE PR 14, Low 1)
    assert extract.last_review_status(conn, "a_aaaaaa_1.jpg") == "kept_hand"
    assert _where(proj)["a_aaaaaa_1.jpg"] == "dropped/unplanned"


def test_filters_leave_kept_hand_alone_and_it_anchors_duplicates():
    rows = [frame("f0", t=0, h="0000000000000000", blur=1, status="kept_hand"),
            frame("f1", t=1, h="0000000000000001", blur=50),
            frame("f2", t=2, h="00000000000000ff", blur=60)]
    st = extract.filter_frames(rows, 50, 4)["status"]
    # f0 is the blurriest but kept by hand: not dropped, not in the output, still the anchor.
    # Blur 50 % of the 2 filtered frames = 1 (f1); f2 is 8 bits from f0: kept.
    assert st == {"f1": "dropped_blur", "f2": "kept"}
    st = extract.filter_frames(rows, 0, 4)["status"]
    assert st == {"f1": "dropped_duplicate", "f2": "kept"}  # 1 bit from kept-by-hand f0


def test_counts_include_kept_hand_and_set_rejected_outside_the_plan_apart(tmp_path):
    proj, conn = _project(tmp_path, [
        frame("k.jpg"), frame("h.jpg", status="kept_hand"), frame("b.jpg", status="dropped_blur"),
        frame("r.jpg", status="rejected"), frame("old.jpg", status="rejected")])
    c = extract.counts(conn, planned={"k.jpg", "h.jpg", "b.jpg", "r.jpg"})
    assert (c["kept"], c["kept_hand"], c["rejected"], c["rejected_unplanned"], c["total"]) == \
        (2, 1, 1, 1, 4)
    assert extract.counts(conn)["total"] == 5  # without the plan, every rejected frame counts
    # the filter preview counts the same way when given the plan (QE PR 14, Low 2)
    p = extract.preview(conn, 0, 0, planned={"k.jpg", "h.jpg", "b.jpg", "r.jpg"})["counts"]
    assert (p["rejected"], p["total"]) == (1, 4)


def test_extract_sets_kept_hand_aside_and_brings_it_back_kept_by_hand(tmp_path):
    src = make_video(tmp_path / "v" / "dive.MP4", seconds=6, size="160x120")
    row = {"path": str(src), "path_code": "abc123", "duration_s": 6.0}
    proj = tmp_path / "p"
    proj.mkdir()
    db.connect(proj / "state.sqlite", create=True).close()
    plan, _ = extract.sample_plan([row], [], 2.0, 300)      # 0, 2, 4 s
    extract.run(proj, plan)
    conn = db.connect(proj / "state.sqlite")
    files = [r[0] for r in conn.execute("SELECT file FROM frames ORDER BY timestamp_ms")]
    review.act(conn, proj, files[2], "D")                   # 4 s rejected
    conn.execute("UPDATE frames SET status='dropped_blur' WHERE file=?", (files[1],))
    extract.move_home(proj, files[1], "dropped_blur")
    review.act(conn, proj, files[1], "K")                   # 2 s kept by hand
    conn.commit()
    small, _ = extract.sample_plan([row], [], 2.0, 1)       # only 0 s
    extract.run(proj, small, blur_percent=50)
    st = dict(conn.execute("SELECT file, status FROM frames"))
    assert st == {files[0]: "kept", files[1]: "unplanned", files[2]: "rejected"}
    extract.run(proj, plan, blur_percent=50)                # all three again, blur 50 %
    st = dict(conn.execute("SELECT file, status FROM frames"))
    assert st[files[1]] == "kept_hand" and st[files[2]] == "rejected"
    assert _where(proj)[files[2]] == "rejected"


def test_review_page_keys_and_stage_through_the_app(tmp_path):
    lib = tmp_path / "lib"
    make_video(lib / "a.MP4", seconds=6, size="1280x720")
    before = digest(lib)
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(lib)]))
    client.post("/projects", data={"name": "p"})
    client.post("/p/p/ingest/add", data={"folder": str(lib), "paths": [str(lib / "a.MP4")]})
    client.post("/p/p/extract/filter", data={"blur": "0", "dup": "0"})
    client.post("/p/p/extract/start", data={"every": "2", "cap": "300"})
    assert wait_for(lambda: client.get("/p/p/extract/status").json()["job"].get("state") == "done")
    folder = data / "projects" / "p"
    conn = sqlite3.connect(folder / "state.sqlite")
    files = [r[0] for r in conn.execute("SELECT file FROM frames ORDER BY timestamp_ms")]
    assert len(files) == 3

    page = client.get("/p/p/review").text
    assert all(f in page for f in files) and 'id="per-row"' in page and "viewer.js" in page
    act = lambda key, f="": client.post("/p/p/review/act", json={"file": f, "key": key})
    assert act("D", files[0]).json()["status"] == "rejected"
    assert (folder / "rejected" / files[0]).is_file()
    assert files[0] not in client.get("/p/p/review").text          # kept only by default
    assert files[0] in client.get("/p/p/review?dropped=1").text    # "Show dropped"
    assert act("Z").json()["status"] == "kept"
    assert (folder / "frames" / files[0]).is_file()
    assert act("D", files[0]).json()["status"] == "rejected"
    assert act("Z").status_code == 200 and act("D", files[0]).status_code == 200
    assert act("Q", files[0]).status_code == 400
    assert client.post("/p/p/review/act", content=b"[]").status_code == 400
    stage = dict(conn.execute("SELECT stage, state FROM stage_status"))
    assert stage["review"] == "in progress"

    # the rejected frame is still served full size and as a grid thumbnail, never cropped
    assert client.get(f"/p/p/frames/{files[0]}").status_code == 200
    t = client.get(f"/p/p/framethumbs/{files[0]}")
    assert t.status_code == 200
    (tmp_path / "t.jpg").write_bytes(t.content)
    with Image.open(tmp_path / "t.jpg") as im:
        assert im.size == (640, 360)
    assert client.get("/p/p/framethumbs/..%2Fstate.sqlite").status_code == 404
    # another page in the browser cannot act (D14)
    other = client.post("/p/p/review/act", json={"file": files[1], "key": "D"},
                        headers={"Origin": "http://evil.example"})
    assert other.status_code == 403

    # the extract counts add up with the rejected frame (D20)
    c = client.get("/p/p/extract/status").json()["counts"]
    assert c["kept"] + c["dropped_blur"] + c["dropped_duplicate"] + c["rejected"] == c["total"] == 3
    # Extract again leaves the rejected frame where it is
    client.post("/p/p/extract/start", data={"every": "2", "cap": "300"})
    assert wait_for(lambda: client.get("/p/p/extract/status").json()["job"].get("state") == "done")
    assert (folder / "rejected" / files[0]).is_file()

    assert client.post("/p/p/review/finish", data={"state": "skipped"}).status_code == 200
    assert dict(conn.execute("SELECT stage, state FROM stage_status"))["review"] == "skipped"
    assert 'value="skipped"' not in client.get("/p/p").text           # no Skip once skipped
    assert client.post("/p/p/review/finish", data={"state": "done"}).status_code == 200
    assert client.post("/p/p/review/finish", data={"state": "x"}).status_code == 400
    assert digest(lib) == before                                      # SPEC constraint 4
