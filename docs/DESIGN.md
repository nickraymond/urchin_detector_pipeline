# DESIGN.md — Architecture & Decisions (as-built)

*What it did / how it's shaped. Agents append; never silently rewrite history.*
*Last updated: 2026-11-25*

## System topology

S1 (a test, not the tool): `uv run sam-test --video <file>` → `app/sam_test/run.py` pulls 3
frames with ffmpeg into `<data_dir>/sam_test/frames/<video>/`, then starts `app/sam_test/worker.py` as a
separate process for each attempt (Apple GPU → Apple GPU with CPU fallback → CPU, stopping at
the first that works). Each worker writes `result_<attempt>.json` after every step;
`report.py` turns them into `<data_dir>/sam_test/<video>/<model>_tile<N>_cutoff<C>/report.html` with
the overlays embedded, plus an all-runs page `<data_dir>/sam_test/report.html`.

## Key designs

- Meta's code is installed unmodified (pinned commit); `device_patch.py` makes the
  run-time changes it needs off-CUDA and lists every one on the report page (D8).
- One process per attempt, so a hard crash in the Apple GPU driver still leaves a report (D9).

### Planned for S2 (frames) — approved by Nick 2026-11-24; becomes as-built as bites land

```
app/config.py            read config.local.toml; stop with a clear message if a path is missing
app/project.py           create / open a project folder under <data_dir>/projects/<name>/
app/db.py                the project's state.sqlite: tables and small read / write helpers
app/stages/ingest.py     find videos and image folders; read details with ffprobe
app/stages/clips.py      playback copies, thumbnail strip, load / save clips.json
app/stages/extract.py    plan the samples, pull frames with ffmpeg, blur and duplicate filters
app/stages/review.py     keep / delete (move to rejected/) / undo
app/web/server.py        the web pages and their actions (FastAPI)
app/web/templates/       plain HTML pages
app/web/static/          one CSS file, small plain JavaScript files
```

- Stage modules are plain functions that take a project and return data; the web layer only
  calls them, so they are tested without a browser.
- The server listens on `127.0.0.1` only. Pages load nothing from the internet (D12).
- Every page and file the server serves keeps the host and origin rule (D14): only the host
  names `127.0.0.1` and `localhost` are answered, and any request other than GET / HEAD must come
  from the app's own page (`Origin` or `Referer` equal to the request's host and port). Every
  response carries `Cross-Origin-Resource-Policy: same-origin`, so no other web page can embed
  a thumbnail or frame. GET requests change nothing in a project; the only files they write
  are caches (thumbnails).
- Ingest (as-built, bite 1b): the scan re-reads the folder on each page load; ffprobe results are
  kept in memory per (path, size, modified time) while the app runs. Only videos the scan of the
  posted folder found can be added, never a path typed into the form. A video that fails the
  one-frame decode check cannot be added. A video whose file name and size are already in the
  project, from any folder, is not added again (D13). At most 200 videos are read per page load;
  a bigger folder asks for a subfolder until bite 3's background jobs exist.
  Each video gets a thumbnail (one frame 10 % in, 240 px wide) in `<data_dir>/thumbs/<path code>.jpg`,
  served only as `/thumbs/<6 hex>.jpg` from that folder (Nick's request on PR 4).
- Video work goes through the installed `ffmpeg` / `ffprobe` as subprocesses; no extra video library.
- Blur score (variance of Laplacian) and near-duplicate hash use numpy / scipy / Pillow, already installed.
- Long steps run in the background behind one small interface (start, progress, done / error) so
  the S4 job-control module can replace it without touching the stages.
- Data: `project.json` (name, created, format version, settings: seconds between samples, frame
  cap, blur threshold, duplicate distance, JPEG quality); `clips.json` (list of
  `{video, start_s, end_s, note}`); `state.sqlite` tables `videos` (path, path code, recording,
  chapter, duration, size, fps, codec), `frames` (file, video, timestamp ms, blur score, hash,
  status), `stage_status` (stage, state, counts), `review_actions` (undo history).
- Frame status values: `kept`, `dropped_blur`, `dropped_duplicate`, `rejected`. Path code: 6
  hex characters made from the video's full path, as the S1 test already does.
- Tests: synthetic videos made by ffmpeg at test time and synthetic images; temporary folders;
  page actions through FastAPI's test client, checked against the database and files on disk.
  No real footage in tests.

## Decision log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-11-24 | Data and local paths live outside git: `config.local.toml` (gitignored) names the video folders and the data folder on the SSD; `config.example.toml` is the committed pattern. | SPEC hard constraints 2–3; Nick chose the SSD (chat 2026-11-24). |
| D2 | 2026-11-24 | Branch names are `sprint/<n>-<name>`; the word "slug" is not used. | Nick, chat 2026-11-24. |
| D3 | 2026-11-24 | SAM 3.1 is tested on the laptop (sprint S1) before any tool code. | The official readme lists a CUDA GPU as a prerequisite; the laptop is an M1 Max. Nick, chat 2026-11-24. |
| D4 | 2026-11-24 | Job control (queue, schedule, pause, log) is a standalone module with no labeling-specific code. | Nick wants to reuse it for nightly training and evaluation jobs. Nick, chat 2026-11-24. |
| D5 | 2026-11-24 | Kelp and kelp shoot classes are deferred; classes stay a settings-file change. | Nick has no reference images yet. Nick, chat 2026-11-24. **Replaced by D11 (2026-11-24).** |
| D6 | 2026-11-24 | The class list starts small (purple urchin, red urchin, sea star, sunflower sea star, anemone, fish, rock/substrate) and grows toward the Reef Check California manual's species list. Class ids are only ever appended, never renumbered or reused. | Prove the pipeline first; splitting a class later costs a re-review of that class only. Stable ids keep old exports and models readable. Nick, chat 2026-11-24. Kelp shoot and kelp were added to the starting list the same day: D11. |
| D7 | 2026-11-24 | Pin PyTorch to 2.10.0 (the version in Meta's sam3 readme), Python 3.12. | torch 2.15.0 aborts on the Apple GPU on macOS 14.6 (`MPSNDArray arrayViewWithShape:strides:` unrecognized selector) on a plain Conv2d; 2.8.0–2.14.0 pass the same test (S1 test, 2026-11-24). |
| D8 | 2026-11-24 | Run Meta's SAM 3 code unmodified; apply small run-time fixes in `app/sam_test/device_patch.py`: stub the triton-only `edt` module, build two caches on the chosen device instead of `"cuda"`, replace the fused bfloat16 `_addmm_activation` with float32 linear + activation (its bfloat16 output gave NaN scores on MPS and a dtype error on CPU), skip empty point/box encodes and `pin_memory` off-CUDA. Other missing ops use PyTorch's CPU fallback; the report lists which. | Smallest change that runs; Meta's files stay untouched and upgradeable. Each fix is from a failure seen in the S1 smoke test. |
| D9 | 2026-11-24 | The SAM test runs each attempt in its own worker process. | The Apple GPU can abort the whole process (seen with torch 2.15.0); the parent must still write the report. |
| D10 | 2026-11-24 | The S1 test defaults to SAM 3 (`facebook/sam3`, `sam3.pt`), 1008 px tiles, and the prompt "urchin" in place of "sea urchin". | SAM 3 gives the same text results as SAM 3.1, and its click/box model loads (3.1 lacks all 309 of those weights). "urchin" kept far more urchins (Bench results). Nick, chat 2026-11-24. n = 3 frames; to re-check in S3's prompt preview. |
| D11 | 2026-11-24 | Kelp is back in the starting classes: kelp shoot (object) and kelp (area). This replaces D5. | The restoration-site videos show kelp (seen in the S1 demo run), so reference frames exist. Prompts are untested; S3 tries them. Nick, chat 2026-11-24. |
| D12 | 2026-11-24 | S2 stack: FastAPI on `127.0.0.1`, plain HTML and plain JavaScript, no internet assets; ffmpeg / ffprobe as subprocesses; a low-res H.264 playback copy for scrubbing. Starting filter values: the blur threshold slider starts at the value that drops the blurriest 10% of the sampled frames (their 10th-percentile score); a small hash distance for duplicates; both adjustable in the UI. "Suggest clips" and "Pre-screen with SAM" are left out of S2. | Boring and debuggable; browser playback of 4K HEVC is unverified; the filter values are starting guesses to tune on real footage, not measurements. Nick approved the S2 plan, chat 2026-11-24. |
| D13 | 2026-11-24 | S2 bite 1 is split into two PRs: 1a (start, settings errors, projects, stage list) and 1b (ingest). Files with the same name and size in two folders count as one video. Settings errors are shown in the browser and the settings file is re-read on every page load; the app never creates `data_dir` (an unplugged SSD would otherwise put data on the laptop disk). Added `python-multipart` (Apache-2.0, PyPI metadata), which FastAPI needs for form posts. | Plan estimate grew to ~450 lines once ingest had to pick files or folders; the SSD holds 17 copied videos. Nick approved, chat 2026-11-24. |
| D14 | 2026-11-24 | Host and origin rule for the web server: `TrustedHostMiddleware` answers only `127.0.0.1` and `localhost`; a request other than GET / HEAD whose `Origin` (or, if absent, `Referer`) is not `http://<the request's own host:port>` gets 403. Every later page, including those that serve frames, stays behind it. Opening a project never creates a missing `state.sqlite`; only project creation does. | Binding 127.0.0.1 alone does not stop another web page in the browser from posting to the app, or from reading pages through DNS rebinding; once frames are served that would let images leave the laptop (SPEC constraint 1). A silently recreated database would lose review state. QE findings 1 and 2 on PR 3, 2026-11-24. |
| D14a | 2026-11-24 | Addition to D14: every response carries `Cross-Origin-Resource-Policy: same-origin`. GET requests change nothing in a project; writing a cache (thumbnails) is their only side effect. Thumbnails are rebuilt when the video at that path changes size or modified time. | Thumbnails are the first images the app serves; without the header any web page could show them in an `<img>` and learn which exist. QE nits 1–3 on PR 4, 2026-11-24. |
| D15 | 2026-11-24 | Playback copy: video track only, scaled to 960 px wide, H.264 from the Mac's built-in encoder (`h264_videotoolbox`) at 2 Mb/s, originals decoded in software; a GoPro `.LRV` is repackaged without re-encoding. Thumbnail strip: one 160 px frame every 5 s, made from the copy. Both in the project folder (`playback/`, `strips/`). Background jobs (`app/jobs.py`) keep their status in memory only; output is written to temporary names, so a stopped job is simply queued again. | Measured on the SSD: 4K 47 s → 10 s, 5.3K 60 fps 153 s → 74 s (39 MB); hardware decode 113 s on 5.3K and failed on 4K; strip from the copy 2 s vs 20 s from the original. Nick chose 960 px and the 2a / 2b split, chat 2026-11-24. |
| D15a | 2026-11-24 | Addition to D15: playback copies start only from the "Make playback copies" button (a POST), never from opening a page. Thumbnail n is the first frame at or after n × 5 s (absolute time). An `.LRV` is repackaged only if ffprobe says it is H.264, else it is encoded like an original; copies are never scaled up. | QE findings 1–3 on PR 5: a GET that starts minutes of CPU work can be triggered by any web page; the old strip filter drifted 0.6 s per 10 min at 29.97 fps (measured; new one stays within one frame); newer GoPro `.LRV` codecs are unknown. |
| D16 | 2026-11-24 | `clips.json` is `{"clips": [{video (full path), start_s, end_s, note}]}`; the page sends one video's whole clip list on every change and the server checks it (end after start, at least 0.5 s, inside the video, no overlap) before an atomic write. Frames per stretch = samples at 0, N, 2N … = ceil(length / N); a video with no clips counts whole. | Overlaps refused rather than merged (Nick approved the plan, chat 2026-11-24); one list per save keeps the file and the page in step. The count is before the blur / duplicate filters and the 300 cap, which bite 3 applies. Clips of a video removed on the Ingest page stay in `clips.json` on purpose: they are not counted or used, and come back if the video is added again (Remove never deletes work). Times must be finite numbers; a `clips.json` of the wrong shape is reported on the page as "cannot be read" (QE findings on PR 6). Each clip row shows its first frame from `/p/<name>/frame/<code>.jpg?t=`, cut from the playback copy and cached in `<project>/clipthumbs/<code>/`; only a saved clip's start time is served, so that cache is bounded by the clip list and grows only through the app's own saves (D14a; QE finding on PR 6). |
| D17 | 2026-11-24 | Extract (bite 3): the blur threshold is per video, not per project: the slider drops the blurriest X % of each video's samples (start 10 %); this replaces the project-wide 10th percentile in D12. Frames are decoded from the original with ffmpeg (seek, raw RGB through a pipe) and saved by Pillow at JPEG quality 95. Dropped frames are moved, never erased, to `<project>/dropped/blurry/` and `dropped/duplicate/`; kept ones stay in `frames/`; applying new filter values moves files back and forth. Bite 3 is split into 3a (plan, extraction, counts) and 3b (filters). | Measured on the SSD: sharpness scores 17–257 on a 5.3K video vs 141–800 on a 4K one, so one project-wide cut-off drops almost only the hazier video; 0.5–0.8 s per full-size frame, ~4 min for 300 frames of 5.3K; dHash distance 8–45 between frames 2 s apart. Nick approved, chat 2026-11-24. |
| D17a | 2026-11-25 | Extract keeps exactly the current plan: frames `kept` by earlier settings that the current plan does not ask for get status `unplanned` and move, never erased, to `<project>/dropped/unplanned/`; they move back if a later plan asks for them. Frames are decoded through a PPM pipe, so the size comes from the image header and rotated videos come out upright. | QE review of PR 7: changing the settings left 12 kept frames for a plan of 3; a rotated video would fail every frame on the raw-RGB size check. |
| D18 | 2026-11-25 | Urchin type is set by hand, not by SAM. SAM runs one prompt, "urchin", which purple urchin and red urchin share in `classes.yaml`; its masks are saved as urchins with no type (no class id) until tagged, and changing the prompt re-runs both classes (SPEC Q7). In label review `P` / `R` tags the selected urchin purple or red; a frame cannot be accepted while it has an untagged urchin. `classes.yaml` and every export keep the two classes purple urchin and red urchin (ids unchanged, D6); an untagged urchin is never exported. | S1 showed SAM cannot tell purple from red: "purple sea urchin" kept 0 where "urchin" kept 39 on one patch (Bench results, n = 3 frames). The hourly packet needs count and cover per type (`docs/source/mvp_spec.md`). Nick approved, chat 2026-11-25. |
| D19 | 2026-11-25 | Filters (bite 3b): per video, blur first: the blurriest `blur_percent` % of a video's frames (rounded down; slider 0–50, start 10) become `dropped_blur`; then in time order a frame whose dHash differs from the previous *kept* frame by fewer than `duplicate_distance` of 64 bits (slider 0–32, start 4, 0 = off) becomes `dropped_duplicate`. `rejected` and `unplanned` frames are left alone. The page previews counts and examples with a read-only GET; Apply (POST) saves both values to `project.json` (`blur_percent` replaces `blur_threshold`) and moves the files at once, no background job; Apply is refused while extraction runs. Each status has a home folder (`frames/`, `dropped/blurry/`, `dropped/duplicate/`, `dropped/unplanned/`); a file found elsewhere is moved home by the next Apply or Extract, and Extract never decodes a planned frame whose file is in any of them. Extract applies the saved values at its end, so kept + blurry + duplicate (+ rejected) = the plan. | Nick approved the 3b plan, chat 2026-11-25. Blur first so a blurry frame never anchors a duplicate check; "fewer than" makes 0 mean off. Moving 44 files took well under a second on the scratch project. QE PR 7 watch item (a planned frame in `dropped/blurry/` must not be decoded again). |
| D19a | 2026-11-25 | Addition to D19: the filter preview shows one row per filter with every frame it would drop, blurry frames by the blur setting at which each is dropped (the cut is per video), then most blurry first, so a stronger blur setting only adds frames on the right; duplicates least unique first (fewest bits differing from the last kept frame), where a stronger setting can also add frames mid-row because the frame each is compared with changes. The kept examples that spec §3 and TRACKER 3b mention are removed, at Nick's request. | Nick, builder chat 2026-11-25: "Too complicated, too much being show"; he asked for one sorted row per filter, then chose this left-to-right order. |
| D20 | 2026-11-25 | **Lead call** (planning lead, CLAUDE.md rule 4): frame review (bite 4) shows kept frames by default. A "Show dropped" toggle adds blurry, duplicate and rejected frames, each with a label naming its status. `K` on any of them gives it the new status `kept_hand` (home `frames/`); `D` moves a frame to `rejected/`; `Z` undoes. This amends D19: the filters leave `kept_hand` frames out (not in `FILTERED`), so the next Apply or Extract never drops them again; a `kept_hand` frame anchors the duplicate check like any kept frame; the kept count includes it. The counts become kept (with `kept_hand`) + dropped blurry + dropped duplicate + rejected = the live count (or the cap); rejected frames outside the current plan stay in `rejected/` and are counted separately. | The spec gives review both `D` delete and `K` keep (`docs/source/mvp_spec.md` §4); `K` only does something if non-kept frames can be seen. The filters are starting guesses (D12), so rescuing a wrongly dropped frame by hand is cheaper than re-tuning a slider. Kept-only by default keeps the normal pass fast. Nick can reverse this. |

## Bench / test results

**S1, 2026-11-24 — SAM 3.1 (`sam3.1_multiplex.pt`) on the M1 Max, Apple GPU + CPU fallback,
torch 2.10.0, 3 frames (4K HEVC) from one Mendocino Headlands ReefCheck video, cut-off 0.5.**
Source: `<data_dir>/sam_test/sam3.1_tile504_cutoff0.5/result_mps-fallback.json`.

| Mode | s / frame | Peak memory | Objects kept (frame 2, ~hundreds of urchins visible) |
|---|---|---|---|
| whole frame (→1008 px) | 3.9 | 7.4 GB process, 14.4 GB Apple GPU | fish 12; no urchins |
| 504 px tiles (60 per frame) | 233 | same | urchin prompts 46–125 each, in a few tiles only; "sea anemone" 91, mostly on urchins |

Model load 8.7 s. Only `aten::_assert_async` fell back to the CPU. The SAM 3.1 checkpoint
supplies none of the 309 weights of the click/box model, so the click test is not valid with
3.1. Visual check of the frame-2 overlay: most urchins missed; the one real anemone not labelled.

**S1, 2026-11-24 — SAM 3 (`sam3.pt`), same frames and set-up.** Sources: run folders
`sam3_tile504_cutoff0.5/` and `sam3_tile1008_cutoff0.5_own-prompts/` under `<data_dir>/sam_test/`.

| Run | s / frame (tiles) | Urchins kept, frames 1 / 2 / 3 (tiles, duplicates at tile edges not removed) |
|---|---|---|
| SAM 3, 504 px tiles, S1 prompt list | 233 | "sea urchin" 6 / 139 / 20 — same patchy result as SAM 3.1 |
| SAM 3, 1008 px tiles, prompt "urchin" | 53 | 5 / 283 / 156; whole frame (3.7 s) 0 / 21 / 19 |

Prompt wording matters more than model or tile size: on one 1000 px patch of frame 2,
"urchin" kept 39 (score ≥ 0.5), "sea urchin" 0, "purple sea urchin" 0, "black sea urchin" 0.
Visual check, frame 2 with "urchin": most urchins on the near rock marked; distant ones and
most of murky frame 1 missed; the green anemone labelled "sea anemone". SAM 3's click model
loads fully (0 of 309 weights missing): in the demo run the click on frame 1's best urchin
gave an outline with score 0.887 (box 0.901), 0.7 s (`<data_dir>/sam_test/sam3_tile1008_cutoff0.5/
result_mps-fallback.json`, run before per-video folders). n = 3 frames from one video; not yet a basis for defaults.
