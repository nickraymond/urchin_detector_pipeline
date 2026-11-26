"""S5 fix (D50): a project made before D39 has no black urchin; the Review page offers to add it
(appended, new id), and nothing else in classes.yaml or the SAM results changes."""

from app.sam import classes
from tests.fake_sam import run_sam
from tests.test_preview import make_project

OLD = classes.DEFAULT_FILE.read_text().replace(
    "  - {id: 10, name: black urchin, kind: object, prompts: [urchin], sam_group: urchin}\n", "")


def old_project(tmp_path, text=OLD):
    client, folder = make_project(tmp_path, 1)
    assert "black urchin" not in text
    (folder / "classes.yaml").write_text(text)
    run_sam(folder, ["v_aaaaaa_0.png"])
    return client, folder


def test_new_project_is_offered_nothing(tmp_path):
    client, folder = make_project(tmp_path, 1)
    run_sam(folder, ["v_aaaaaa_0.png"])
    assert classes.missing_types(folder, classes.ensure(folder)["classes"]) == []
    assert "labels/add-type" not in client.get("/p/p/labels").text


def test_old_project_adds_black_urchin_at_the_end(tmp_path):
    client, folder = old_project(tmp_path)
    before = classes.parse((folder / "classes.yaml").read_text())["classes"]
    raw = sorted(p.read_bytes() for p in (folder / "sam_raw").glob("[!.]*.json"))
    page = client.get("/p/p/labels").text
    assert "no <b>black urchin</b> type" in page and "class id\n    10" in page
    r = client.post("/p/p/labels/add-type", data={"type": "black urchin"}, follow_redirects=False)
    assert r.status_code == 303 and "error" not in r.headers["location"]
    after = classes.parse((folder / "classes.yaml").read_text())["classes"]
    assert after[:-1] == before                              # appended only
    assert after[-1] == {"id": 10, "name": "black urchin", "kind": "object",
                         "prompts": ["urchin"], "sam_group": "urchin"}
    assert sorted(p.read_bytes() for p in (folder / "sam_raw").glob("[!.]*.json")) == raw
    assert "labels/add-type" not in client.get("/p/p/labels").text
    v = client.get("/p/p/labels/frame/v_aaaaaa_0.png").json()
    assert ("black urchin", "B") in [(t["name"], t["key"]) for t in v["types"]]
    # adding it twice is refused and changes nothing
    r = client.post("/p/p/labels/add-type", data={"type": "black urchin"}, follow_redirects=False)
    assert "not+a+missing" in r.headers["location"]
    assert classes.parse((folder / "classes.yaml").read_text())["classes"] == after


def test_a_used_id_10_gets_a_new_id_and_a_renamed_type_counts(tmp_path):
    text = OLD.replace("  - {id: 2, name: red urchin,", "  - {id: 10, name: crab, kind: object, "
                       "prompts: [crab]}\n  - {id: 2, name: red urchin,")
    client, folder = old_project(tmp_path, text)
    cl = classes.ensure(folder)["classes"]
    (new,) = classes.missing_types(folder, cl)
    assert new["name"] == "black urchin" and new["id"] == max(c["id"] for c in cl) + 1
    # a type renamed in the project (same id) is not offered again
    renamed = [c for c in cl if c["name"] != "crab"] + [{**new, "id": 10, "name": "dark urchin"}]
    assert classes.missing_types(folder, renamed) == []
    r = client.post("/p/p/labels/add-type", data={"type": "black urchin"}, follow_redirects=False)
    assert "error" not in r.headers["location"]
    assert classes.parse((folder / "classes.yaml").read_text())["classes"][-1]["id"] == new["id"]


def test_a_removed_type_with_results_gets_back_the_id_its_results_hold(tmp_path):
    """QE PR 58: black urchin had results as id 10, then was removed; a new id would be refused
    by the used-id check (D6), so the id the results hold is offered."""
    import json
    client, folder = old_project(tmp_path)
    used = folder / "sam_raw" / classes.USED_FILE
    used.write_text(json.dumps({**json.loads(used.read_text()), "10": "black urchin"}))
    (new,) = classes.missing_types(folder, classes.ensure(folder)["classes"])
    assert new["id"] == 10
    r = client.post("/p/p/labels/add-type", data={"type": "black urchin"}, follow_redirects=False)
    assert "error" not in r.headers["location"]
    assert classes.parse((folder / "classes.yaml").read_text())["classes"][-1]["id"] == 10
