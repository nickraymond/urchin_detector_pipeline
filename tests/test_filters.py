"""S2 bite 3b: blur and near-duplicate filters, Apply, and extraction applying them.
Frames are tiny made-up files or synthetic video; no real footage."""

import sqlite3

from app import db, project
from app.stages import extract
from tests.test_app_skeleton import client_for, write_config
from tests.test_clips import wait_for
from tests.test_ingest import digest, make_video

H0 = "0000000000000000"


def frame(file, video="/v/a.MP4", t=0, blur=100.0, h=H0, status="kept"):
    return {"file": file, "video": video, "timestamp_ms": t, "blur_score": blur, "hash": h,
            "status": status}


def test_blur_is_per_video_and_rounds_down():
    rows = [frame(f"a{i}", t=i, blur=10 * i + 1, h=f"{1 << i:016x}") for i in range(10)]
    rows += [frame(f"b{i}", video="/v/b.MP4", t=i, blur=1000 + i, h=f"{1 << i:016x}")
             for i in range(5)]
    st = extract.filter_frames(rows, 20, 0)["status"]
    # 20 % of 10 = 2 from a; 20 % of 5 = 1 from b, though every b frame is sharper than any a
    assert sorted(f for f, s in st.items() if s == "dropped_blur") == ["a0", "a1", "b0"]
    assert set(extract.filter_frames(rows, 0, 0)["status"].values()) == {"kept"}


def test_duplicates_chain_on_the_last_kept_frame_after_blur():
    rows = [frame("f0", t=0, h="0000000000000000"),
            frame("f1", t=1, h="0000000000000007", blur=1),   # blurriest: dropped first
            frame("f2", t=2, h="0000000000000003"),           # 2 bits from f0
            frame("f3", t=3, h="000000000000000f"),           # 4 bits from f0, 2 from f2
            frame("f4", t=4, h="00000000000000ff")]           # 8 bits from f0
    st = extract.filter_frames(rows, 20, 4)["status"]
    # f1 never anchors; f3 compares with f0 (the last *kept* frame), not with dropped f2
    assert st == {"f0": "kept", "f1": "dropped_blur", "f2": "dropped_duplicate",
                  "f3": "kept", "f4": "kept"}
    assert extract.filter_frames(rows, 20, 0)["status"]["f2"] == "kept"   # 0 = off


def test_rejected_and_unplanned_are_left_alone():
    rows = [frame("r", t=0, blur=0, status="rejected"),
            frame("u", t=1, blur=0, status="unplanned"),
            frame("k", t=2, blur=50)]
    assert extract.filter_frames(rows, 50, 64)["status"] == {"k": "kept"}


def _project(tmp_path, rows):
    """A project with the given frame rows, each with a small file in its status folder."""
    proj = tmp_path / "p"
    proj.mkdir()
    conn = db.connect(proj / "state.sqlite", create=True)
    for r in rows:
        folder = proj / "rejected" if r["status"] == "rejected" else extract.home(proj, r["status"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / r["file"]).write_bytes(b"jpeg " + r["file"].encode())
        conn.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", (
            r["file"], r["video"], r["timestamp_ms"], r["blur_score"], r["hash"], r["status"]))
    conn.commit()
    return proj, conn


def _where(proj):
    return {p.name: str(p.parent.relative_to(proj)) for p in proj.rglob("*.jpg")}


def test_apply_moves_back_and_forth_and_counts_add_up(tmp_path):
    rows = [frame(f"a_aaaaaa_{i}.jpg", t=i, blur=i, h=f"{(1 << (i % 3)) - 1:016x}")
            for i in range(10)]
    rows.append(frame("a_aaaaaa_99.jpg", t=99, status="rejected"))
    proj, conn = _project(tmp_path, rows)
    preview = extract.preview(conn, 30, 2)
    c = extract.apply_filters(conn, proj, 30, 2)
    assert c == {**preview["counts"], "unplanned": 0, "stills": 0, "stills_rejected": 0}  # = preview
    assert c["dropped_blur"] == 3 and c["dropped_duplicate"] > 0
    assert c["kept"] + c["dropped_blur"] + c["dropped_duplicate"] + c["rejected"] == c["total"] == 11
    where = _where(proj)
    assert len(where) == 11  # nothing erased
    for f, st in conn.execute("SELECT file, status FROM frames"):
        if st != "rejected":
            assert where[f] == extract.HOMES[st], f
    assert where["a_aaaaaa_99.jpg"] == "rejected"
    c = extract.apply_filters(conn, proj, 0, 0)  # back again: every frame comes home
    assert c["kept"] == 10 and len(list(extract.frames_dir(proj).glob("*.jpg"))) == 10


def test_preview_lists_every_dropped_frame_in_order(tmp_path):
    """Nick 2026-11-25: one row per filter, everything it drops; most blurry and least unique
    on the left, so a stronger setting adds frames on the right."""
    rows = [frame("a_aaaaaa_0.jpg", t=0, blur=50, h="0000000000000000"),
            frame("a_aaaaaa_1.jpg", t=1, blur=5),                        # blurriest of a
            frame("a_aaaaaa_2.jpg", t=2, blur=60, h="0000000000000001"),  # 1 bit from a_0
            frame("a_aaaaaa_3.jpg", t=3, blur=70, h="0000000000000007"),  # 3 bits from a_0
            frame("b_bbbbbb_0.jpg", video="/v/b.MP4", t=0, blur=20),     # blurriest of b
            frame("b_bbbbbb_1.jpg", video="/v/b.MP4", t=1, blur=100, h="0000000000000000"),
            frame("b_bbbbbb_2.jpg", video="/v/b.MP4", t=2, blur=110, h="0000000000000003"),
            frame("b_bbbbbb_3.jpg", video="/v/b.MP4", t=3, blur=120, h="00000000000000ff")]
    proj, conn = _project(tmp_path, rows)
    p = extract.preview(conn, 25, 4)
    assert [e["blur"] for e in p["blur_dropped"]] == [5, 20]               # most blurry first
    assert [(e["file"], e["distance"]) for e in p["dup_dropped"]] == [
        ("a_aaaaaa_2.jpg", 1), ("b_bbbbbb_2.jpg", 2), ("a_aaaaaa_3.jpg", 3)]  # least unique first
    assert p["counts"] == {"kept": 3, "kept_hand": 0, "dropped_blur": 2, "dropped_duplicate": 3,
                           "rejected": 0, "total": 8}
    assert _where(proj) == {r["file"]: "frames" for r in rows}     # preview moved nothing


def test_a_stronger_blur_setting_only_adds_frames_on_the_right(tmp_path):
    """QE PR 15: the cut is per video, so raw sharpness order could put a newly dropped frame
    in the middle of the row; the row follows the setting at which each frame goes."""
    rows = [frame(f"a_aaaaaa_{i}.jpg", t=i, blur=10 * (i + 1)) for i in range(5)]  # 10..50
    rows += [frame(f"b_bbbbbb_{i}.jpg", video="/v/b.MP4", t=i, blur=100 + i) for i in range(10)]
    rows.append(frame("a_aaaaaa_9.jpg", t=9, blur=1, status="kept_hand"))  # never ranked (D20)
    proj, conn = _project(tmp_path, rows)
    before = []
    for pct in range(0, 51):
        row = [e["file"] for e in extract.preview(conn, pct, 0)["blur_dropped"]]
        assert row[:len(before)] == before, pct
        before = row
    assert len(before) == 7  # 2 of a's 5 and 5 of b's 10 at 50 %


def test_a_stop_mid_apply_is_repaired(tmp_path):
    rows = [frame("a_aaaaaa_0.jpg", t=0, blur=1), frame("a_aaaaaa_1.jpg", t=1, blur=50)]
    proj, conn = _project(tmp_path, rows)
    # stopped after the file moved to dropped/blurry/, before its row changed
    extract.move_home(proj, "a_aaaaaa_0.jpg", "dropped_blur")
    c = extract.apply_filters(conn, proj, 0, 0)
    assert c["kept"] == 2 and _where(proj) == {r["file"]: "frames" for r in rows}


def test_extraction_applies_the_filters_and_never_decodes_a_dropped_frame(tmp_path,
                                                                           monkeypatch):
    src = make_video(tmp_path / "v" / "dive.MP4", seconds=10, size="160x120")
    before = digest(tmp_path / "v")
    row = {"path": str(src), "path_code": "abc123", "duration_s": 10.0}
    proj = tmp_path / "p"
    proj.mkdir()
    db.connect(proj / "state.sqlite", create=True).close()
    plan, live = extract.sample_plan([row], [], 2.0, 300)   # 5 frames
    assert extract.run(proj, plan, blur_percent=40, duplicate_distance=0) == 5
    conn = sqlite3.connect(proj / "state.sqlite")
    c = extract.counts(conn)
    assert c["dropped_blur"] == 2 and c["kept"] + c["dropped_blur"] + c["dropped_duplicate"] \
        == c["total"] == live
    blurry = [r[0] for r in conn.execute("SELECT file FROM frames WHERE status='dropped_blur'")]
    assert all((extract.home(proj, "dropped_blur") / f).is_file() for f in blurry)
    # QE PR 7 watch item: a planned frame in dropped/blurry/ counts as done
    monkeypatch.setattr(extract, "grab", lambda *a: (_ for _ in ()).throw(AssertionError))
    assert extract.run(proj, plan, blur_percent=40) == 0
    assert extract.counts(conn)["dropped_blur"] == 2
    # a later plan without them sets the dropped frames aside too
    assert extract.run(proj, plan[:1], blur_percent=0) == 0
    c = extract.counts(conn)
    assert c["total"] == 1 and c["unplanned"] == 4
    assert digest(tmp_path / "v") == before  # SPEC constraint 4


def test_filter_routes(tmp_path):
    lib = tmp_path / "videos"
    make_video(lib / "a.MP4", seconds=6, size="160x120")
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data, roots=[str(lib)]))
    client.post("/projects", data={"name": "p"})
    client.post("/p/p/ingest/add", data={"folder": str(lib), "paths": [str(lib / "a.MP4")]})
    client.post("/p/p/extract/start", data={"every": "1", "cap": "300"})
    assert wait_for(lambda: client.get("/p/p/extract/status").json()["job"].get("state") == "done")
    page = client.get("/p/p/extract").text
    assert 'name="blur"' in page and 'value="10"' in page  # default 10 %: 6 frames -> none
    folder = data / "projects" / "p"
    files_before = sorted(p.name for p in folder.rglob("*.jpg"))

    a = client.get("/p/p/extract/filter?blur=50&dup=0").json()
    assert a["counts"] == {"kept": 3, "kept_hand": 0, "dropped_blur": 3, "dropped_duplicate": 0, "rejected": 0,
                           "total": 6}
    assert len(a["blur_dropped"]) == 3 and a["dup_dropped"] == []
    assert [e["blur"] for e in a["blur_dropped"]] == sorted(e["blur"] for e in a["blur_dropped"])
    for bad in ("blur=51&dup=0", "blur=nan&dup=0", "blur=10&dup=4.5", "blur=10&dup=abc"):
        assert client.get(f"/p/p/extract/filter?{bad}").status_code == 400, bad
    assert sorted(p.name for p in folder.rglob("*.jpg")) == files_before  # preview reads only

    assert client.post("/p/p/extract/filter", data={"blur": "50", "dup": "0"},
                       headers={"Origin": "http://evil.example"}).status_code == 403
    r = client.post("/p/p/extract/filter", data={"blur": "50", "dup": "0"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/p/p/extract/status").json()["counts"]["dropped_blur"] == 3
    assert len(list((folder / "dropped" / "blurry").glob("*.jpg"))) == 3
    settings = project.open_project(data, "p")["settings"]
    assert settings["blur_percent"] == 50 and settings["duplicate_distance"] == 0
    assert "blur_threshold" not in settings
    dropped = a["blur_dropped"][0]["file"]
    assert client.get(f"/p/p/frames/{dropped}").status_code == 200  # examples can be shown
    assert sorted(p.name for p in folder.rglob("*.jpg")) == files_before  # moved, not erased
