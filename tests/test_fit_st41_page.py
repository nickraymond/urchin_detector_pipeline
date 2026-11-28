"""S6 bite 2f, page part: the N6's second compiler (ST Edge AI Core 4.1.0, D71) on its own line with its own
button; never a cell's headline, never queued by "Run all". Fake 8-bit step and fake compiler, no ST tool."""

from app import modeldefs
from app.fit import convert, table
from tests.test_fit_page import convert_jobs, placed, site, wait_all_done, write_fit  # noqa: F401

KEY = "plainseg-s-256-c6"


def test_the_second_line_is_on_n6_cells_only_and_does_not_set_the_headline(tmp_path):
    write_fit(tmp_path, "yolox-nano-conv-256-c5", "n6_st41", converts=True, placement=placed(117, 2, 0),
              tool="ST Edge AI Core v4.1.0-20732", note="placement only")
    before = len(table.missing(tmp_path))
    cells = {r["kind"]: r for r in table.table(tmp_path)}["detector"]["cells"]
    yolo = next(c for c in cells["n6"]["candidates"] if c["key"] == "yolox-nano-conv-256-c5")
    assert yolo["code"] == "not_tried" and cells["n6"]["code"] == "not_tried"        # 4.0.0 has no result yet
    assert yolo["second"]["code"] == "runs" and yolo["second"]["step"] == "n6_st41"
    assert all(c["second"] is None for cam in ("pi", "ae3") for c in cells[cam]["candidates"])
    assert before == len(modeldefs.candidates()) * 3                                 # Run all: never the 4.1.0 step


def test_page_shows_the_line_and_its_button(site):  # noqa: F811
    client, data = site
    write_fit(data, "rfdetr-nano-384-c5", "n6_st41", converts=False, stage="compile",
              error="TOOL ERROR: tuple index out of range", tool="ST Edge AI Core v4.1.0-20732")
    page = client.get("/fit").text
    assert page.count("↳ ST Edge AI Core 4.1.0</td>") == len(modeldefs.candidates())
    assert page.count("are a second N6 compiler beside the known-good 4.0.0") == len(modeldefs.KINDS)
    assert 'name="camera" value="n6_st41"' in page and "TOOL ERROR: tuple index out of range" in page
    assert "Run all (" + str(len(modeldefs.candidates()) * 3) + " cells not tried)" in page


def test_its_button_runs_the_step_and_run_again_on_the_n6_queues_it_with_the_others(site):  # noqa: F811
    client, data = site
    folder = data / "fit" / KEY
    r = client.post("/fit/run", data={"key": KEY, "camera": "n6_st41"}, follow_redirects=False)
    assert "OpenMV+N6+%28ST+Edge+AI+Core+4.1.0%29+queued." in r.headers["location"]
    client.post("/fit/run", data={"key": KEY, "camera": "n6"}, follow_redirects=False)
    wait_all_done(client, data)
    first = convert.read_fit(folder, "n6_st41")
    assert first and convert.read_fit(folder, "n6")
    r = client.post("/fit/run", data={"key": KEY, "camera": "n6"}, follow_redirects=False)
    assert "queued+again" in r.headers["location"] and "ST+Edge+AI+Core+4.1.0%29+with+it" in r.headers["location"]
    assert len(wait_all_done(client, data)) == 4                    # two first runs, the redo, the 4.1.0 step again
    again = convert.read_fit(folder, "n6_st41")
    assert again["when"] >= first["when"] and again["int8"] == convert.read_fit(folder, "n6")["int8"]
    assert (folder / "int8" / "calls.log").read_text() == "int8\n"


def test_an_older_4_1_result_says_run_again(tmp_path):
    arch = next(a for a in modeldefs.candidates() if modeldefs.arch_key(a) == "yolox-nano-conv-256-c5")
    old = {**convert.made_with(arch), "code": {**arch["code"], "commit": "an older commit"}}
    write_fit(tmp_path, "yolox-nano-conv-256-c5", "n6_st41", converts=True, placement=placed(117, 2, 0), made_with=old)
    cells = {r["kind"]: r for r in table.table(tmp_path)}["detector"]["cells"]
    yolo = next(c for c in cells["n6"]["candidates"] if c["key"] == "yolox-nano-conv-256-c5")
    assert yolo["second"]["older"] and not yolo["older"]
