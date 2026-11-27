"""S8 bite 2a: the Datasets page and the freeze job, on a real export (fake SAM, synthetic
frames) made through the app's own pages, so the version reads exactly what Export writes."""

import json

from tests.test_clips import wait_for
from tests.test_export_page import accept_all
from tests.test_review_page import labeled


def test_freeze_from_the_page_on_a_real_export(tmp_path):
    client, folder = labeled(tmp_path)
    page = client.get("/p/p/datasets").text
    assert "No export yet" in page and "No version yet." in page
    assert 'href="/p/p/datasets"' in client.get("/p/p").text
    r = client.post("/p/p/datasets/freeze", data={}, follow_redirects=False)
    assert r.status_code == 303 and "No+export+yet" in r.headers["location"]
    accept_all(client)
    client.post("/p/p/export/start")
    assert wait_for(lambda: "Last export" in client.get("/p/p/export").text, 60)
    # refused while the export job is still finishing, accepted once it has ended
    def freeze():
        return client.post("/p/p/datasets/freeze", data={"version_name": "first try"},
                           follow_redirects=False).headers["location"]
    seen = []
    assert wait_for(lambda: seen.append(freeze()) or seen[-1] == "/p/p/datasets", 60)
    assert all("export+is+queued+or+running" in s for s in seen[:-1])
    assert wait_for(lambda: "Version 1: first try" in client.get("/p/p/datasets").text, 60)
    page = client.get("/p/p/datasets").text
    m = json.loads((folder / "datasets" / "v1" / "manifest.json").read_text())
    assert m["hash"][:12] in page and "purple urchin" in page and "splits from the export" in page
    assert len(m["frames"]) == 2 and sum(m["counts"]["frames"].values()) == 2
    assert {c["name"] for c in m["classes"]} >= {"unknown", "purple urchin", "kelp"}
    for f in m["frames"]:
        assert (folder / "datasets" / "images" / f["image"]).is_file()
        assert (folder / "datasets" / "v1" / "seg_masks" / f["seg_mask"]).is_file()
    assert "Freeze a dataset version" in client.get("/jobs").text
    # the same export again: the job says so and no version 2 appears
    client.post("/p/p/datasets/freeze", data={})
    assert wait_for(lambda: "nothing changed since version 1" in client.get("/p/p/datasets").text, 60)
    assert not (folder / "datasets" / "v2").exists()
    # a name that is too long is refused before any job
    r = client.post("/p/p/datasets/freeze", data={"version_name": "x" * 81}, follow_redirects=False)
    assert "too+long" in r.headers["location"]
