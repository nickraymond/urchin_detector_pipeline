# DEV_LOG.md — Session Log

*Newest entries on top. One entry per working session. Short: what changed,
what broke, what's next.*

---

## Entry template

```
## YYYY-MM-DD — Sprint Sn — <one-line summary>
**Branch:** sprint/n-name
**Done:**  <bullets>
**Broke/surprised us:** <bullets or "nothing">
**Next:** <the single next bite>
```

---

## 2026-11-24 — Sprint S2 — Bite 2b: clip marks (I / O), clips.json, counts
**Branch:** sprint/2-frames-2b (from main `9b296cd`, after PR 5 merged)
**Done:** plan approved by Nick. Timeline under the player with clip bands and a playhead
(click to jump); keys I / O / Space / arrows (Shift = 5 s), ignored while typing a note; clip list
with note, Go to, Start = now, End = now, Delete; every change saved at once to `clips.json`
(atomic). Server checks each video's clips: order, overlap, at least 0.5 s, inside the video.
Counts per video and in total at one frame every 2 s (whole video when unmarked); Pick clips row
on the project page shows "N clips, M:SS marked". 4 new tests (61 in total). Browser check with
real key presses on a real video: I at 0:05, O at 0:15 made the clip; an overlapping clip was
refused with its message; a note survived a reload; Delete returned the video to "whole video".
**Broke/surprised us:** Nick's 2a test found an empty strip: macOS writes `._` shadow files next
to every file on the SSD (fixed in PR 5).
**Next:** QE reviews the draft PR; Nick runs the 2b manual test; then bite 3 (extract frames).

## 2026-11-24 — Sprint S2 — Bite 2a: playback copies and player
**Branch:** sprint/2-frames-2a (from main `4d11e30`)
**Done:** bite 2 plan approved by Nick: split 2a (player) / 2b (marks), copies 960 px wide.
`app/jobs.py`: one background worker, jobs queued and run one at a time with percent and time
left. `app/stages/clips.py`: 960 px H.264 playback copy (Mac's built-in encoder; a GoPro .LRV
is repackaged instead) and a strip of one thumbnail every 5 s made from the copy, both written to
temporary names first. Pick clips page: video list with progress, the player, the strip (click to
jump). 6 new tests (55 in total). Browser check on 2 real 4K videos: the copy was made in the
background, the page reloaded when ready, clicking the 0:20 thumbnail jumped the player to 20 s.
**Broke/surprised us:** measured before coding: decoding the originals with the Mac's video chip
was slower on 5.3K and failed on 4K, so originals are decoded in software. GoPro files carry a
data track that has to be left out of the copy.
**Next:** QE reviews the draft PR; Nick runs the 2a manual test; then bite 2b (I / O marks).

## 2026-11-24 — Sprint S2 — Bite 1b: ingest page
**Branch:** sprint/2-frames-1b (from main `b00be85`, after PR 3 merged)
**Done:** plan approved by Nick. `app/stages/ingest.py`: find .MP4 / .MOV under a folder and its
subfolders, skip `._` files, .LRV / .THM and hidden folders; ffprobe details plus a one-frame
decode check; GoPro chapters grouped; same name + size counted once ("also in"). Ingest page with
a folder browser (starts at `video_roots`, or a pasted path), tick boxes, add / remove; the
project's Ingest row shows "done · N videos (M recordings)". 12 new tests (43 in total), one
proving source files are byte-identical with the same modified time afterwards.
On the SSD: 29 videos listed, 17 with "also in", none failing the decode check. The first page
load takes ~6.6 s (ffprobe + decode of 29 videos); later loads ~0.1 s (cached while the app runs).
Nick's manual test (2 videos added) worked; he asked for a Finder-style thumbnail per video:
added (one frame 10 % in, 240 px, cached in `<data_dir>/thumbs/`); the first load that builds all
29 took ~16 s, later loads reuse them.
**Broke/surprised us:** of two copies, the one in the folder that sorts first is the one listed.
`uv run app` right after stopping a copy said "port in use" (closing connections); fixed.
**Next:** Nick runs the 1b manual test; QE reviews the draft PR; then bite 2 (clip picker).

## 2026-11-24 — Sprint S2 — Bite 1a: app skeleton (start, settings errors, projects)
**Branch:** sprint/2-frames
**Done:** bite 1 plan approved by Nick, split into 1a (skeleton) and 1b (ingest); same-name,
same-size videos will count as one. `uv run app` serves on 127.0.0.1:8000 and opens the dashboard.
Settings problems show as a red box in the browser and the settings file is re-read on every
page load. Projects: create / open / list under `<data_dir>/projects/`, with `project.json` and
`state.sqlite`; the project page lists the six stages. 14 new tests (31 in total) pass.
**Broke/surprised us:** FastAPI form posts need `python-multipart` (Apache-2.0), which the
plan did not list. The SSD's video folder holds 46 videos but only 29 distinct ones: 17 are
copies in a second folder. No GoPro-named files there, so chapter grouping can only be tested on
made-up names.
**Next:** Nick runs the 1a manual test; QE reviews the draft PR; then bite 1b (ingest).

## 2026-11-24 — Sprint S2 — Plan written (planning lead)
**Branch:** sprint/2-frames
**Done:** S1 merged to main as `a362495` (QE PASS on `db29464`; Nick ran the demo). S2 plan
approved by Nick: five bites in TRACKER, planned design in DESIGN, D12. Kelp and kelp shoot are
starting classes again (D11). SPEC Q1, Q6, Q13 and part of Q10 answered.
**Broke/surprised us:** the S1 demo video had almost no urchins but did have kelp.
**Next:** S2 bite 1, plan nibble — Nick gate.

## 2026-11-24 — Sprint S1 — SAM test runner built; Apple GPU works with run-time fixes
**Branch:** sprint/1-sam-test
**Done:** plan approved by Nick. `pyproject.toml` (uv, Python 3.12, torch 2.10.0, Meta's sam3
pinned at `31cb7e6`). `app/sam_test/`: frame pull, one worker process per attempt, report page
with overlays embedded, click and box test. 9 unit tests on synthetic images (13 by the end of S1). Smoke test with
untrained weights: text, click and box prompts run on the Apple GPU (~2.3 s per 1008 px image).
Dry run on a real video: frames extracted; report shows the "download the weights" error.
**Broke/surprised us:** torch 2.15.0 aborts on the Apple GPU on macOS 14.6 → pinned 2.10.0 (D7).
Meta's image code needs six run-time fixes off-CUDA plus PyTorch's CPU fallback (D8). The bite
is ~630 lines of app code, about twice the plan's ~340, because of the fixes and the worker.
Later the same day: Nick's first real run found nothing — my bfloat16 replacement gave NaN
scores on the Apple GPU (fixed, CPU and GPU now agree). Added `--tile` and `--cutoff`, one folder
per setting combination and an all-runs index page. SAM 3.1 at 504 px tiles: 233 s per frame,
finds urchins only in patches and calls many urchins "sea anemone" (DESIGN, Bench results).
The Hugging Face `LICENSE` for `facebook/sam3.1` matches the GitHub one word for word.
Then (Nick approved the download): SAM 3 gives the same text results as 3.1, but its click
model works (clean urchin outline from one click). The prompt word "urchin" finds far more than
"sea urchin" (DESIGN, Bench results). Added `--prompts`. Licence of `facebook/sam3` = GitHub text.
Nick chose SAM 3 + "urchin" as defaults and to close S1 (D10). Smaller crops do not help
mid-range urchins (504 → 4, 336 → 2, 252 → 0 kept on one hazy patch).
**Next:** Nick runs the S1 demo; QE reviews the draft PR; then S2 plan.

## 2026-11-24 — Sprint S0 — Spec review edits, QE fixes, SAM test sprint added
**Branch:** sprint/0-scaffold
**Done:** Nick ran the S0 demo (3 checks passed). QE gave PASS-with-nits on 51dabf1; its 12
findings are fixed or answered. Ten spec edits from Nick's review written to
`docs/source/mvp_spec.md`. New sprint S1 "SAM test on the laptop" added; later sprints moved
down one. `uv` installed; Nick logged in to Hugging Face and confirmed access to `facebook/sam3.1`.
**Broke/surprised us:** `brew install hf uv` started building `llvm` and `rust` from source on
macOS 14 — stopped; use the standalone `uv` installer and `uvx hf`. The SAM 3 readme lists a
CUDA GPU as a prerequisite and does not mention Apple chips.
**Next:** S1, plan nibble — Nick gate.

## 2026-11-24 — Pre-work — Project scaffolded

**Branch:** n/a
**Done:** docs created from `template_LLM_project`; SPEC and sprint ladder filled from
`docs/source/mvp_spec.md`; `.gitignore` and `config.example.toml` added. Planning session, no app code.
**Broke/surprised us:** nothing yet.
**Next:** Nick runs the S0 demo; then S1 plan nibble — Nick gate.
