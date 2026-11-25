"""S3 bite 6: the class editor form (SPEC Q17). The form sends {settings, classes} as JSON;
classes.yaml is written only when the same checks as a text save pass. Synthetic only."""

import copy
import json
import random

import pytest

from app.sam import classes
from tests.test_app_skeleton import client_for, write_config

DEFAULT = classes.parse(classes.DEFAULT_FILE.read_text())


def start(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data))
    client.post("/projects", data={"name": "p"})
    return client, data / "projects" / "p"


def with_results(folder, used):
    (folder / "sam_raw").mkdir(exist_ok=True)
    (folder / "sam_raw" / classes.USED_FILE).write_text(json.dumps(used))


def test_form_saves_an_added_class_and_check_writes_nothing(tmp_path):
    client, folder = start(tmp_path)
    page = client.get("/p/p/labeling").text
    assert 'id="class-form-data"' in page and "classform.js" in page and "<textarea" not in page
    form = json.loads(page.split('id="class-form-data">')[1].split("</script>")[0])
    assert form["next_id"] == 10 and form["kinds"] == ["object", "area"] and form["used"] == {}
    body = {"settings": form["settings"], "classes": form["classes"] + [
        {"id": 10, "name": "sea cucumber", "kind": "object", "prompts": ["sea cucumber"]}]}
    r = client.post("/p/p/labeling/classes/form?check=1", json=body)
    assert r.status_code == 200 and r.json() == {"errors": []}
    assert not (folder / "classes.yaml").exists()                    # a check never writes
    assert client.post("/p/p/labeling/classes/form", json=body).json() == {"errors": []}
    saved = classes.parse((folder / "classes.yaml").read_text())
    assert saved["classes"][-1]["name"] == "sea cucumber"
    assert "next free id: 11" in client.get("/p/p/labeling").text


def test_problems_point_at_the_row_or_setting_and_nothing_is_saved(tmp_path):
    client, folder = start(tmp_path)
    body = copy.deepcopy(DEFAULT)
    body["classes"][4]["name"] = "Sea Star"                    # row 5 repeats row 3's name
    body["classes"][1]["prompts"] = ["red urchin"]             # group needs the same prompts
    body["settings"]["tile_px"] = 100
    r = client.post("/p/p/labeling/classes/form", json=body)
    assert r.status_code == 400
    errs = r.json()["errors"]
    assert any(e["row"] == 4 and "used twice" in e["msg"] for e in errs)
    assert any(e["row"] == 1 and "same prompts" in e["msg"] for e in errs)
    assert any(e["setting"] == "tile_px" for e in errs)
    assert not (folder / "classes.yaml").exists()
    for bad in (None, [], {"settings": [], "classes": []}, {"classes": []}):
        r = client.post("/p/p/labeling/classes/form", json=bad)
        assert r.status_code == 400 and r.json()["errors"][0]["row"] is None
    assert client.post("/p/p/labeling/classes/form", content=b"{oops",
                       headers={"content-type": "application/json"}).status_code == 400
    assert not (folder / "classes.yaml").exists()


def test_a_class_with_results_keeps_its_name(tmp_path):
    client, folder = start(tmp_path)
    with_results(folder, {"6": "fish"})
    page = client.get("/p/p/labeling").text
    assert json.loads(page.split('id="class-form-data">')[1].split("</script>")[0])["used"] == {
        "6": "fish"}
    body = copy.deepcopy(DEFAULT)
    body["classes"][5]["name"] = "rockfish"
    errs = client.post("/p/p/labeling/classes/form", json=body).json()["errors"]
    assert any(e["row"] == 5 and "never reused" in e["msg"] for e in errs)
    del body["classes"][5]                                    # removing it is allowed
    assert client.post("/p/p/labeling/classes/form", json=body).json() == {"errors": []}
    body["classes"].append({"id": 10, "name": "fish", "kind": "object", "prompts": ["fish"]})
    errs = client.post("/p/p/labeling/classes/form", json=body).json()["errors"]
    assert any(e["row"] == len(body["classes"]) - 1 and "under id 6" in e["msg"] for e in errs)


# --- fuzz: no form input can produce a classes.yaml that fails the checks ---

TEXTS = ["urchin", "", " ", "sea star", "yes", "no", "null", "~", "true", "1", "0x10", "1e3",
         "a: b", "#x", "- y", "[z]", "{w}", "'q'", '"d"', "&a", "*a", "!t", "%x", "@x", "`x`",
         "a\nb", "tab\t", "é ü 海胆", "kelp", "rock/substrate", "x" * 300, " ", "\x00",
         "\x85", "---", "..."]
VALUES = [None, True, False, 0, 1, -1, 2, 9, 10, 255, 0.5, 1.5, 1e300, [], {}, [1], {"a": 1},
          *TEXTS[:12]]


def mutate(rng, body):
    b = copy.deepcopy(body)
    for _ in range(rng.randint(1, 4)):
        cl, st = b["classes"], b["settings"]
        ds = [c for c in cl if isinstance(c, dict)]
        op = rng.randrange(11)
        if op == 0 and ds:
            rng.choice(ds)["name"] = rng.choice(TEXTS + VALUES)
        elif op == 1 and ds:
            rng.choice(ds)["prompts"] = rng.choice([[rng.choice(TEXTS)], [], rng.choice(VALUES),
                                                    [rng.choice(TEXTS), rng.choice(VALUES)]])
        elif op == 2 and ds:
            rng.choice(ds)["kind"] = rng.choice(["object", "area", *VALUES])
        elif op == 3 and ds:
            rng.choice(ds)["sam_group"] = rng.choice(TEXTS + VALUES)
        elif op == 4 and ds:
            rng.choice(ds)["seg_as"] = rng.choice(TEXTS + VALUES)
        elif op == 5 and ds:
            rng.choice(ds)["id"] = rng.choice(VALUES + [rng.randint(-3, 30)])
        elif op == 6:
            st[rng.choice(list(classes.SETTINGS) + ["extra"])] = rng.choice(
                VALUES + [rng.uniform(-5, 5000), rng.randint(0, 5000)])
        elif op == 7 and cl:
            cl.pop(rng.randrange(len(cl))) if rng.random() < .5 else rng.shuffle(cl)
        elif op == 8 and st:
            del st[rng.choice(list(st))]
        elif op == 9 and cl:
            cl[rng.randrange(len(cl))] = rng.choice(VALUES)        # not a class at all
        else:
            cl.append({"id": rng.randint(1, 30), "name": rng.choice(TEXTS),
                       "kind": rng.choice(["object", "area"]), "prompts": [rng.choice(TEXTS)],
                       **({rng.choice(["sam_group", "seg_as", "colour"]): rng.choice(TEXTS)}
                          if rng.random() < .4 else {})})
    return b


@pytest.mark.parametrize("seed", range(4))
def test_no_form_input_saves_a_file_that_fails_the_checks(tmp_path, seed):
    rng = random.Random(seed)
    folder = tmp_path / "proj"
    folder.mkdir()
    with_results(folder, {"3": "sea star", "9": "kelp"})
    classes.form_save(folder, DEFAULT)
    saved_ok = refused = 0
    for _ in range(300):
        body = mutate(rng, DEFAULT)
        before = (folder / "classes.yaml").read_bytes()
        errors = classes.form_save(folder, body)
        after = (folder / "classes.yaml").read_text()
        data = classes.parse(after, classes.used_ids(folder))   # the file always passes
        if errors:
            refused += 1
            assert after.encode() == before                     # a refused save writes nothing
            assert all(set(e) == {"row", "setting", "msg"} and e["msg"] for e in errors)
        else:
            saved_ok += 1
            assert data == {"settings": body["settings"], "classes": body["classes"]}
            classes.form_save(folder, DEFAULT)                  # back to the start
    assert saved_ok > 10 and refused > 100
