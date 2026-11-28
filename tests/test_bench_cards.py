"""S7 bite 2a (D84): test cards as data and pages, plus QE PR 98's nits 1 and 2. Fake unit and made-up names only;
every test uses its own temporary data folder (QE PR 98 nit 3)."""

import json

import pytest
from fastapi.testclient import TestClient

from app import config
from app.bench import card, riglog, units
from app.web import server


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    for key in ("yolox_nano", "yolox_tiny"):         # S6's layout: every candidate's file has the same name
        (d / "fit" / key / "int8").mkdir(parents=True)
        (d / "fit" / key / "int8" / "model_int8.tflite").write_bytes(b"model bytes " + key.encode())
    (d / "other.tflite").write_bytes(b"x")
    return d


def form(**over):
    base = {"question": "How fast is YOLOX Nano on the fake N6?", "unit": "fake", "board": "n6",
            "models": "fit/yolox_nano/int8/model_int8.tflite", "windows": "3", "n": "300", "cost_minutes": "5",
            "pass_bar": "median within 10 % of 10.6 ms", "stop_rules": "\n".join(card.DEFAULT_STOP_RULES),
            "written_by": "lead"}
    return {**base, **over}


def test_build_save_load_review(data):
    c = card.build(data, units.Bench(), form())
    assert c["models"][0]["name"] == "yolox_nano" and len(c["models"][0]["sha256"]) == 64
    assert "RESTORE.md matches the rig log" in c["stop_rules"]                 # D87
    two = card.build(data, units.Bench(), form(models="fit/yolox_nano/int8/model_int8.tflite\n"
                                                      "fit/yolox_tiny/int8/model_int8.tflite\nother.tflite"))
    assert [m["name"] for m in two["models"]] == ["yolox_nano", "yolox_tiny", "other"]
    n = card.save(data, c)
    assert n == 1 and card.save(data, c) == 2                                  # never overwrites
    got = card.load(data, 1)
    assert not got["reviewed"] and got["review"] is None
    with pytest.raises(card.CardError, match="other than its writer"):
        card.review(data, 1, "Lead")
    assert card.review(data, 1, "qe")["reviewed"]
    with pytest.raises(card.CardError, match="already reviewed"):
        card.review(data, 1, "someone")
    (card.folder(data) / "._3.json").write_bytes(b"\0\x05AppleDouble")       # macOS sidecar on the exFAT disk
    assert [x["number"] for x in card.list_cards(data)] == [2, 1]
    assert card.save(data, c) == 3


def test_card_changed_after_review_is_not_reviewed(data):
    card.save(data, card.build(data, units.Bench(), form()))
    card.review(data, 1, "qe")
    p = card.folder(data) / "1.json"
    p.write_text(p.read_text().replace("10.6 ms", "20 ms"))
    got = card.load(data, 1)
    assert got["review"] and not got["reviewed"]


@pytest.mark.parametrize("over, why", [
    ({"question": " "}, "the question is empty"),
    ({"windows": "1"}, "at least 3"),
    ({"windows": "x"}, "at least 3"),
    ({"n": "0"}, "inferences per window"),
    ({"cost_minutes": ""}, "the cost"),
    ({"models": ""}, "at least one model file"),
    ({"models": "/etc/hosts"}, "not under the data folder"),
    ({"models": "../outside.tflite"}, "not under the data folder"),
    ({"models": "fit/yolox_nano/int8/missing.tflite"}, "model file not found"),
    ({"models": "fit/yolox_nano/int8/model_int8.tflite\nfit/yolox_nano/int8/model_int8.tflite"}, "two model files"),
    ({"unit": "bench-a"}, "not a bench unit"),
    ({"board": "imx708"}, "has no imx708 board"),
    ({"stop_rules": "\n"}, "at least one stop rule"),
    ({"pass_bar": ""}, "the pass bar is empty"),
])
def test_bad_card_is_refused_with_its_reason(data, over, why):
    with pytest.raises(card.CardError, match=why):
        card.build(data, units.Bench(), form(**over))
    assert not card.folder(data).exists()


def test_field_unit_card_is_refused(data):
    bench = units.parse({"field_units": ["field-pi"], "units": {"bench-a": {"ssh_host": "h", "cameras": ["n6"]}}})
    with pytest.raises(card.CardError, match="field unit"):
        card.build(data, bench, form(unit="Field-Pi"))
    assert card.build(data, bench, form(unit="bench-a"))["unit"] == "bench-a"


def test_field_count_counts_entries_as_written():                            # QE PR 98 nit 1
    b = units.parse({"field_units": ["field-pi.local", "Field-Pi.local", "cam-2"]})
    assert b.field_count == 2 and "field-pi" in b.field_units


def test_after_with_no_before_is_shown(tmp_path):                            # QE PR 98 nit 2
    with riglog.change(tmp_path, "bench-a", "pi", "copy x", "delete x"):
        pass
    with open(riglog.path(tmp_path), "a") as f:
        f.write(json.dumps({"change": "zz", "phase": "after", "ok": True, "unit": "bench-a", "at": "t"}) + "\n")
    lines = riglog.path(tmp_path).read_text().splitlines()
    with open(riglog.path(tmp_path), "a") as f:
        f.write(lines[1] + "\n")                                             # a second "after" for a change
    rows = riglog.read(tmp_path)
    assert [r["state"] for r in rows] == ["unmatched", "unmatched", "done"]
    assert "no \"before\"" in rows[1]["what"] and rows[1]["unit"] == "bench-a"


@pytest.fixture
def client(tmp_path, data):
    p = tmp_path / "config.local.toml"
    p.write_text(f'data_dir = "{data}"\nvideo_roots = []\n')
    app = server.create_app(p)
    yield TestClient(app, base_url="http://127.0.0.1:8000", headers={"origin": "http://127.0.0.1:8000"})
    app.state.close_jobs()


def test_cards_pages(client, data):
    page = client.get("/bench/cards").text
    assert "No test cards yet" in page and "RESTORE.md matches the rig log" in page
    r = client.post("/bench/cards", data=form(windows="2"), follow_redirects=True)
    assert "at least 3" in r.text and not card.folder(data).exists()
    r = client.post("/bench/cards", data=form(), follow_redirects=True)
    assert r.url.path == "/bench/cards/1" and "Card 1 saved" in r.text and "not reviewed yet" in r.text
    r = client.post("/bench/cards/1/review", data={"reviewed_by": "lead"}, follow_redirects=True)
    assert "other than its writer" in r.text
    r = client.post("/bench/cards/1/review", data={"reviewed_by": "qe"}, follow_redirects=True)
    assert "Review recorded" in r.text and "Record the review" not in r.text
    assert "qe" in client.get("/bench/cards").text
    assert client.get("/bench/cards/9").status_code == 404
    assert 'href="/bench/cards"' in client.get("/").text
