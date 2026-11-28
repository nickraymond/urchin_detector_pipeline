"""A Pi card's thread count (1-4, default 1) is chosen on the form, saved, shown, and reaches the agent (lead, card 3)."""

from pathlib import Path

import pytest

from app.bench import card, units
from app.bench.board import Model, open_board
from app.bench.pi_board import PiBoard
from app.bench.transport import LocalTransport
from app.bench.worker import board_settings
from tests.test_bench_cards import form

BENCH = {"field_units": ["field-pi"], "units": {"bench-a": {"ssh_host": "bench-a", "cameras": ["n6"]}}}


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    (d / "fit" / "yolox_nano" / "int8").mkdir(parents=True)
    (d / "fit" / "yolox_nano" / "int8" / "model_int8.tflite").write_bytes(b"model")
    return d


def test_the_form_sets_threads_on_a_pi_card_only(data):
    b = units.Bench()
    assert card.build(data, b, form(board="pi"))["threads"] == 1                   # default
    assert card.build(data, b, form(board="pi", threads="4"))["threads"] == 4
    for bad in ("0", "5", "two"):
        with pytest.raises(card.CardError, match="threads must be"):
            card.build(data, b, form(board="pi", threads=bad))
    assert "threads" not in card.build(data, b, form(board="n6", threads="4"))     # a camera has no thread count


def test_a_saved_cards_threads_reach_the_agent(data, tmp_path):
    bench = units.parse(BENCH)
    c = card.load(data, card.save(data, card.build(data, bench, form(unit="bench-a", board="pi", threads="3"))))
    assert c["threads"] == 3
    unit = bench.unit("bench-a")
    settings = board_settings(c, unit)
    assert settings == {"threads": 3}
    real = open_board(unit, "pi", data, bench=bench, log_dir=tmp_path / "run", **settings)   # nothing is started
    assert isinstance(real, PiBoard) and real.threads == 3
    local = PiBoard(units.Bench().unit("fake"), tmp_path / "d", LocalTransport(tmp_path / "agent"), threads=settings["threads"])
    try:                                                                         # the same number, through the agent
        assert local.load(Model("m", Path(c["models"][0]["path"])))["threads"] == 3
        assert local.time("m", 2)["threads"] == 3
    finally:
        local.close()
    assert board_settings(c, units.Bench().unit("fake")) == {}                   # the fake board takes no threads
