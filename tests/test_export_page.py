"""S5 bite 6: the Export page and the export job (on the job module), on synthetic frames labeled
by the fake SAM and reviewed through the Review page's routes."""

from tests.test_clips import wait_for
from tests.test_review_page import F, labeled


def accept_all(client, n=2):
    for i in range(n):
        f = f"v_aaaaaa_{i}.png"
        v = client.get(f"/p/p/labels/frame/{f}").json()
        if v["untagged"]:
            client.post(f"/p/p/labels/act/{f}", json={"action": "class", "mask_ids": v["untagged"],
                                                       "class_id": 1})
        assert client.post(f"/p/p/labels/act/{f}", json={"action": "accept"}).status_code == 200


def test_export_page_runs_the_job_and_shows_the_checks(tmp_path):
    client, folder = labeled(tmp_path)
    page = client.get("/p/p/export").text
    assert "No export yet." in page and "<b>0</b> accepted frames" in page
    r = client.post("/p/p/export/start", follow_redirects=False)
    assert r.status_code == 303 and "No+accepted+frames" in r.headers["location"]
    accept_all(client)
    assert client.post("/p/p/export/start", follow_redirects=False).status_code == 303
    wait_for(lambda: "Last export" in client.get("/p/p/export").text, 60)
    page = client.get("/p/p/export").text
    assert "PASS" in page and "COCO loads with pycocotools" in page and "mask sizes and ids" in page
    assert "fair split" in page                     # one video: the D43 warning
    assert (folder / "export" / "detector_coco.json").is_file()
    assert (folder / "export" / "summary.json").is_file()
    assert 'href="/p/p/export"' in client.get("/p/p").text
    assert 'href="/p/p/export"' in client.get("/p/p/labels").text
    jobs = client.get("/jobs").text
    assert "Export" in jobs


def test_a_failed_export_newer_than_the_good_one_is_reported(tmp_path):
    """writers.run leaves a failed build in export.failed/ (its summary.json and checks.json);
    the page shows the newer of the two, so a FAIL is never hidden by an older PASS."""
    import json
    client, folder = labeled(tmp_path)
    accept_all(client)
    client.post("/p/p/export/start")
    wait_for(lambda: "Last export" in client.get("/p/p/export").text, 60)
    good = json.loads((folder / "export" / "summary.json").read_text())
    failed = folder / "export.failed"
    failed.mkdir()
    (failed / "summary.json").write_text(json.dumps({**good, "ok": False, "at": "2999-01-01T00:00:00"}))
    (failed / "checks.json").write_text(json.dumps(
        [{"name": "mask sizes and ids", "ok": False, "detail": "1 problem(s): a.png: 10x10"}]))
    page = client.get("/p/p/export").text
    assert "FAIL" in page and "was <b>not</b> replaced" in page and "a.png: 10x10" in page
