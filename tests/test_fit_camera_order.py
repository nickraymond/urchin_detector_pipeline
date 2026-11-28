"""Every table on /fit shows the cameras as Pi, N6, AE3 (Nick, 2026-11-27, through the lead), from one shared order."""

import re

from app import modeldefs
from app.fit import convert, table
from tests.test_fit_page import placed, site, write_fit  # noqa: F401

ORDER = ("pi", "n6", "ae3")
LABELS = ("Pi Zero 2W", "OpenMV N6", "OpenMV AE3")


def in_order(text: str, words) -> bool:
    at = [text.find(w) for w in words]
    return -1 not in at and at == sorted(at)


def test_one_shared_order():
    assert convert.CAMERA_ORDER == ORDER and table.CAMERA_ORDER is convert.CAMERA_ORDER
    assert tuple(convert.CAMERAS) == ORDER and tuple(convert.CAMERAS.values()) == LABELS


def test_every_table_on_the_page_is_pi_n6_ae3(site):  # noqa: F811
    client, data = site
    for camera in ORDER:     # a result on each camera, so "stored together" has a best for each
        write_fit(data, "plainseg-s-256-c6", camera, converts=True, file_bytes=1000, limit_bytes=1 << 24,
                  placement=placed(9, 0, 0, "operators"), **({"memory_limit_bytes": 512 << 20} if camera == "pi" else {}))
    page = client.get("/fit").text
    head = re.search(r"<tr><th></th>(.*?)</tr>", page).group(1)                   # the main table's columns
    assert in_order(head, LABELS)
    for kind in modeldefs.KINDS:                                                  # the cell details, per kind
        assert in_order(page, [f'id="{kind}-{c}"' for c in ORDER])
    together = page[page.index('id="together"'):page.index("AE3 time box")]       # "stored together"
    assert in_order(together, LABELS)
    assert table.missing(data)[:3] == [(table.missing(data)[0][0], c) for c in ORDER]   # Run all's queue too
