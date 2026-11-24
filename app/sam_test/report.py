"""Write the one-page S1 report. Images are embedded in the page; nothing loads from the internet."""

from __future__ import annotations

import base64
import html
import time
from pathlib import Path

from app.sam_test.tiles import COLOURS

CSS = """body{font:15px/1.45 -apple-system,Helvetica,sans-serif;margin:24px;max-width:1500px;
color:#111;background:#fff}h1{font-size:22px}h2{font-size:18px;margin-top:32px}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:4px 8px;
text-align:right}th{background:#eee}td:first-child,th:first-child{text-align:left}
.ok{color:#070;font-weight:bold}.bad{color:#b00;font-weight:bold}img{max-width:100%;
border:1px solid #999}pre{background:#f4f4f4;padding:8px;overflow-x:auto;font-size:12px;
white-space:pre-wrap}.key span{display:inline-block;margin-right:14px}
.sw{display:inline-block;width:14px;height:14px;vertical-align:middle;margin-right:4px}"""


def _img(path: Path) -> str:
    if not path.is_file():
        return f"<p class=bad>missing image {html.escape(path.name)}</p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(path.name)}" src="data:image/jpeg;base64,{data}">'


def _pre(text) -> str:
    return f"<pre>{html.escape(str(text))}</pre>" if text else ""


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return f"{sum(vals) / len(vals):.1f}" if vals else "–"


def summary_rows(results: list[dict]) -> str:
    rows = []
    for r in results:
        frames = r.get("frames_done", [])
        status = r.get("status", "?")
        cls = "ok" if status == "ok" else "bad"
        rows.append(
            f"<tr><td>{html.escape(r['attempt'])}</td><td>{html.escape(r['device'])}</td>"
            f"<td>{'on' if r.get('mps_fallback') else 'off'}</td><td class={cls}>{status}</td>"
            f"<td>{r.get('load_seconds', '–')}</td><td>{r.get('warmup_seconds', '–')}</td>"
            f"<td>{_mean([f['whole']['seconds'] for f in frames])}</td>"
            f"<td>{_mean([f['tiles']['seconds'] for f in frames])}</td>"
            f"<td>{r.get('peak_process_gb', '–')}</td><td>{r.get('peak_gpu_gb', '–')}</td>"
            f"<td>{len(frames)}</td></tr>")
    return ("<table><tr><th>attempt</th><th>device</th><th>CPU fallback</th><th>status</th>"
            "<th>model load s</th><th>warm-up s (not counted)</th><th>s / frame, whole</th>"
            "<th>s / frame, tiles</th><th>peak memory GB (process)</th>"
            "<th>peak Apple GPU GB</th><th>frames done</th></tr>" + "".join(rows) + "</table>")


def attempt_section(r: dict, out_dir: Path, prompts: list[str]) -> str:
    parts = [f"<h2>Attempt: {html.escape(r['attempt'])}</h2>"]
    if r.get("status") != "ok":
        parts.append(f"<p class=bad>Error: {html.escape(str(r.get('error')))}</p>")
        parts.append(_pre(r.get("traceback")) + _pre(r.get("stderr_tail")))
    if r.get("patches"):
        parts.append("<p>Changes made at run time to Meta's code (its files are not edited):</p>"
                     "<ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in r["patches"]) + "</ul>")
    if r.get("fallback_ops"):
        parts.append("<p>Steps PyTorch ran on the CPU instead of the Apple GPU: "
                     + html.escape(", ".join(r["fallback_ops"])) + "</p>")
    if r.get("load_messages"):
        parts.append("<p>Messages while loading the checkpoint:</p>" + _pre(r["load_messages"]))
    for f in r.get("frames_done", []):
        parts.append(f"<h3>{html.escape(f['name'])}</h3><table><tr><th>mode</th><th>seconds</th>"
                     "<th>tiles</th>" + "".join(f"<th>{html.escape(p)}</th>" for p in prompts)
                     + "</tr>")
        for mode in ("whole", "tiles"):
            m = f[mode]
            parts.append(f"<tr><td>{mode}</td><td>{m['seconds']}</td><td>{m['tiles']}</td>"
                         + "".join(f"<td>{m['masks'].get(p, 0)}</td>" for p in prompts) + "</tr>")
        parts.append("</table>")
        for mode in ("whole", "tiles"):
            parts.append(f"<p><b>{mode}</b></p>" + _img(out_dir / f[mode]["overlay"]))
    ct = r.get("click_test")
    if ct:
        parts.append("<h3>Click and box prompts (frame 1)</h3>")
        if "error" in ct:
            parts.append(f"<p class=bad>{html.escape(ct['error'])}</p>" + _pre(ct.get("traceback")))
        else:
            parts.append(
                f"<table><tr><th>prompt</th><th>seconds</th><th>score</th><th>result</th></tr>"
                f"<tr><td>one click</td><td>{ct['click']['seconds']}</td>"
                f"<td>{ct['click']['score']}</td><td>{ct['click']['mask_pixels']} mask pixels</td></tr>"
                f"<tr><td>one box</td><td>{ct['box']['seconds']}</td><td>{ct['box']['score']}</td>"
                f"<td>{ct['box']['mask_pixels']} mask pixels</td></tr>"
                f"<tr><td>box as example (finds similar)</td><td>{ct['box_as_example']['seconds']}"
                f"</td><td>–</td><td>{ct['box_as_example']['objects_found']} objects</td></tr></table>"
                f"<p>Click at {ct['click_point']}, box {ct['box_prompt']} (frame pixels), chosen "
                f"from: {html.escape(ct['click_source'])}.</p>")
            parts.append(_pre(ct.get("messages")) + _img(out_dir / ct["overlay"]))
    return "\n".join(parts)


def write_report(out_dir: Path, video_name: str, model: str, prompts: list[str],
                 results: list[dict]) -> Path:
    palette = prompts + ["similar to box", "click", "box"]
    key = "".join(f'<span><i class=sw style="background:rgb{COLOURS[i % len(COLOURS)]}"></i>'
                  f"{html.escape(p)}</span>" for i, p in enumerate(palette))
    ok = any(r.get("status") == "ok" for r in results)
    verdict = ("<p class=ok>SAM ran on this laptop.</p>" if ok else
               "<p class=bad>SAM did not complete on any attempt. Errors are below.</p>")
    body = (f"<h1>S1 SAM test: {html.escape(model)}</h1>"
            f"<p>Video: {html.escape(video_name)} · written {time.strftime('%Y-%m-%d %H:%M')}"
            " · 'whole' = the frame shrunk to 1008 px by SAM; 'tiles' = full size, 1008 px tiles "
            "with 20 % overlap (no duplicate removal across tile edges yet).</p>"
            + verdict + summary_rows(results) + f"<p class=key>{key}</p>"
            + "".join(attempt_section(r, out_dir, prompts) for r in results))
    page = out_dir / "report.html"
    page.write_text(f"<!doctype html><html><head><meta charset=utf-8><title>S1 SAM test</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    return page
