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
- Review (as-built, bite 4, D20): `app/stages/review.py`. `D` / `K` / `Z` post JSON to
  `/p/<name>/review/act`; each change moves the file first, then its row, and adds one
  `review_actions` row; `Z` undoes the newest step whose frame is still as that step left it and removes its row;
  a step whose frame Extract changed since is skipped and kept (it brings a kept-by-hand frame back). `D` on a rejected frame and `K` on
  a kept one change nothing and add no undo step. `rejected` has a home folder, `rejected/`. A
  `kept_hand` frame that a new plan leaves out is set aside in `dropped/unplanned/` like a kept one
  and comes back `kept_hand` (its last review action says so). The grid shows 640 px copies cached
  in `<project>/framethumbs/` (a GET writing a cache, D14a); the full-screen viewer is shared
  with the Extract page (`static/viewer.js`). The dashboard row and the page have Skip; the page
  has Done.
- Stills and resume (as-built, bite 5): the Ingest page also lists the `.JPG` / `.JPEG` / `.PNG`
  images under the open folder; "Add still images" copies them byte for byte into `frames/` as
  `<stem>_<path code>_0.<jpg|png>` (originals only read), records them in a `stills` table (path,
  name, size; same name + size = already added, as D13) and adds each as a `kept` frame with no
  timestamp. A frame with no timestamp is a still: Extract and the filters never touch it, and the
  counts show stills apart from the plan; review treats it like any frame. Resume: every stage
  keeps its state on disk (`state.sqlite`, `clips.json`, files); Extract writes "k of N frames" to
  the dashboard after each frame, and after a stop or a quit its page says how far it got and
  that pressing Extract carries on without decoding finished frames again.
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

### Planned for S3 (SAM on single frames) — lead's plan 2026-11-25 (CLAUDE.md rule 7); becomes as-built as bites land

```
app/sam/default_classes.yaml  the starting class list, copied into a project on first use
app/sam/classes.py            load, check and save a project's classes.yaml
app/sam/worker.py             SAM in its own process (D9): load the model once per job, take frames + prompts,
                              return masks; reuses app/sam_test/device_patch.py unchanged
app/sam/tiles.py              tile layout (moved from app/sam_test/tiles.py, which stays as a re-export so
                              the S1 test and its imports keep working), merge to frame coordinates,
                              tile-edge duplicates (D23)
app/sam/results.py            read / write sam_raw/<frame>.json; settings key per run unit; what must re-run
app/sam/overlap.py            overlap rules (D22); pure numpy, no torch, so tests stay fast
app/stages/label.py           preview and run: pick frames, drive the worker through app/jobs.py, save per frame
app/web/templates/label.html  the Labeling page
```

- `classes.yaml` (per project, next to `project.json`). Settings plus a list of classes. Each class has a fixed
  `id` (append-only, never renumbered or reused, D6; 0 = unknown), `name`, `kind` (`object` or `area`), one or
  more `prompts`, and optionally `sam_group` (classes in one group share one SAM run and its masks carry no
  class id until tagged: purple and red urchin, group `urchin`, D18) and `seg_as` (the area class whose id its
  pixels get in the class map: kelp shoot → kelp, D22). A **run unit** is a `sam_group`, or a class with none.
  Starting ids: 1 purple urchin, 2 red urchin, 3 sea star, 4 sunflower sea star, 5 anemone, 6 fish,
  7 rock/substrate, 8 kelp shoot, 9 kelp (D6 order, then D11). The file lists kelp before rock/substrate
  (D22). Prompts for kelp shoot and kelp are untested; bite 3 tries them. Settings: `confidence` 0.4 (spec),
  `tile_px` 1008 (D10), `tile_overlap` 0.2 and `tile_above_px` 2000 (spec), `min_mask_area_px` (set in bite 3, D24).
- Checks before any save or run: unique ids and names; an id already used in `sam_raw/` keeps its name;
  every class has a kind and at least one prompt; classes of one `sam_group` are objects with the same prompts;
  `seg_as` names an area class. An error names the line and nothing is saved.
- `sam_raw/<frame stem>.json`, written atomically after each frame, also when nothing is found: frame name,
  upright width and height, model, and per run unit: `settings_key` (hash of model, prompts, confidence, tile
  settings, minimum area), the settings and prompts themselves, time taken, and `masks` — each with `prompt`,
  `score`, `box` [x0, y0, x1, y1] in frame pixels, `area`, the mask as COCO RLE of the full frame (pycocotools,
  already a dependency), `class_id` (null inside a `sam_group`), `reviewed` (false until S5) and
  `from_older_settings`.
- Re-run rule (SPEC Q7): a run unit runs on a frame only if its saved `settings_key` differs or is missing.
  On a re-run its unreviewed masks are replaced; reviewed ones stay, marked `from_older_settings`. A class
  removed from `classes.yaml` keeps its saved results (not shown; back if the class returns). Before Run the
  page lists which classes will run and which are unchanged.
- SAM runs in one worker process per job (D9, D26), queued through `app/jobs.py` so only one SAM job runs at a
  time (two runs on the GPU triple the time, S1). Run frames in name order; a stop or quit loses at most the
  frame in progress, and Run again carries on. No pause, schedule or limits yet: that is S4.
- Classes (as-built, bite 1): `app/sam/classes.py`. A page load only reads: a project without
  `classes.yaml` shows the starting file, and the first Save (or, from bite 2, Run) copies it in.
  Save writes `yaml.safe_dump` pieces in the starting file's layout (settings one per line, one class
  per line); comments are not kept. "An id used in `sam_raw/` keeps its name" reads
  `sam_raw/class_ids.json` (`{id: name}` of every class with results), which the SAM runner keeps
  up to date (bite 2); a name with results also keeps its id. Unknown keys and settings are errors
  (typos). Colours come from a fixed palette by id; untagged urchins are grey. The upright helper is
  `app/images.py`; review-grid copies moved to `<project>/framethumbs/upright/`, so the sideways S2
  copies are never served again.
- SAM runner (as-built, bite 2): `app/stages/label.py` `run()` starts `python -m app.sam.worker <job.json>`
  (one process and one model load per call; the app itself never imports torch) and reads its
  progress lines (`@@ {json}` on stdout); the worker's other output goes to `<project>/sam_logs/run-<time>.log`.
  A frame is tiled when its longer side is above `tile_above_px`. Object masks: tile-edge rule then IoU
  (D23, `app/sam/tiles.py`), then the minimum area. Area masks (rock, kelp): all pieces of one prompt
  become one mask, since tiles cut an area into pieces; prompts stay apart in `sam_raw/`. Files are
  `sam_raw/<frame file name>.json` (with the extension, so a still's `.jpg` and `.png` of one stem never
  share one); each run unit records `classes` ({id: name}), and `sam_raw/class_ids.json` collects them.
- Prompt preview (as-built, bite 3): `label.preview_frames` picks first / middle / last kept frame by name
  ("Other frames": 3 at random, D26); `run_preview` is a normal run on those frames (results in `sam_raw/`)
  and writes the timing to `<project>/sam_preview.json`. The page polls `/labeling/status`, shows seconds per
  frame, model load, peak process and Apple GPU memory and a Run estimate, and draws each frame at 1600 px
  (`/labeling/overlay/<file>?hide=`): areas first and faint, objects on top, untagged urchins grey; a toggle per
  run unit. Counts from older settings say so. Kelp prompts and the minimum area come from the bench below.
- Run and overlap (as-built, bite 4): `label.run_all` labels every kept frame in one worker process; Stop
  sets an event that ends the worker, and Run again skips units whose settings key is current. A job may
  return its final message (`app/jobs.py`). `label.run_plan` lists, per run unit, the frames it will run
  on; the estimate is a fixed cost per frame with work plus a cost per unit (`label.frame_costs`, from the
  preview frames' `sam_raw` timings: on 4K frames about 21 s per frame and 5 s per unit; the fixed part
  includes the image encoding, which the first unit pays). `app/sam/overlap.py`: `objects` (rule 1, IoU from
  RLEs) and `class_map` (rules 2-7; an untagged urchin paints 255). The preview's "final class map" draws
  it at 1600 px.
- Large viewer and class legend (as-built, bite 5, SPEC Q16): the Labeling page loads S2's `viewer.js`
  unchanged in its controls; a frame entry may now carry `src` (the overlay URL) and `caption`, and
  `viewerShow(i, true)` redraws without leaving the zoom. `labeling.js` keeps one legend state (hidden run
  units, final class map) for the tiles and the viewer: class buttons toggle, "only" shows one unit (again:
  all); the viewer's legend counts that frame's masks, the page's legend the preview frames together; both
  say objects are boxed and areas are not. The viewer asks for `size=large` (3200 px, never scaled up).
  Speed: `results.decode_box` decodes a mask's box columns only (a vectorised reading of COCO's compressed
  RLE; large boxes use pycocotools' C decode), also used by `overlap.class_map`; the scaled frame, the
  scaled masks and the class-map layer are cached per file version (time and size). Synthetic 5312×2988
  frame with 150 object masks and one area: a toggle took 1.9 s before; now 0.46–0.51 s the first time,
  0.01 s (1600 px) to 0.05 s (3200 px) after.
- Class editor form (as-built, bite 6, SPEC Q17; replaces D25's text box): `static/classform.js` draws one
  row per class from the page's JSON (id and colour shown, never typed; kind, group and "pixels count as"
  are drop-downs of valid values; prompts are chips; ↑ ↓ set the order, which decides which area wins;
  "+ Add class" takes the next id above every id in the file, in `sam_raw/class_ids.json` and given out on
  the page). A class with results has a read-only name with the reason (D6); removing it keeps its results.
  Joining a group copies the group's prompts and makes the class an object, and a group's prompts are
  edited together, so a group never disagrees. After each change the page posts the form to
  `/labeling/classes/form?check=1`; Save posts it again without `check`. Both run `classes.form_check`:
  the form goes through `dump` and then `parse`, the same checks as a text save, and each problem is mapped
  back to its row or setting (`_dump_lines` records where each line came from); nothing is written unless
  it passes. `classes.yaml` stays the file on disk: a file broken in a text editor shows its problems with
  line numbers and the text, without the form, until it is fixed and Reloaded. The old text POST route
  stays (tests use it) but the page no longer has a text box.
- Images are read through one helper that applies the EXIF orientation (Pillow `ImageOps.exif_transpose`), so
  SAM, the masks and the page all use the same upright image (D27).
- Tests: a fake predictor stands in for SAM (no torch, no weights, no real frames); synthetic images; the
  builder runs real SAM in the browser for the self-test and the bench numbers.

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
| D20 | 2026-11-25 | **Lead call** (planning lead, CLAUDE.md rule 4): frame review (bite 4) shows kept frames by default. A "Show dropped" toggle adds blurry, duplicate and rejected frames, each with a label naming its status. `K` on any of them gives it the new status `kept_hand` (home `frames/`); `D` moves a frame to `rejected/`; `Z` undoes. This amends D19: the filters leave `kept_hand` frames out (not in `FILTERED`), so the next Apply or Extract never drops them again; a `kept_hand` frame anchors the duplicate check like any kept frame; the kept count includes it. The counts become kept (with `kept_hand`) + dropped blurry + dropped duplicate + rejected = the live count (or the cap); rejected frames outside the current plan stay in `rejected/` and are counted separately. **Lead call** (planning lead, QE PR 14): a `kept_hand` frame that a new plan leaves out is set aside like `kept` under D17a (to `dropped/unplanned/`) and comes back as `kept_hand` if a later plan asks for it again. | The spec gives review both `D` delete and `K` keep (`docs/source/mvp_spec.md` §4); `K` only does something if non-kept frames can be seen. The filters are starting guesses (D12), so rescuing a wrongly dropped frame by hand is cheaper than re-tuning a slider. Kept-only by default keeps the normal pass fast. Nick can reverse this. |
| D21 | 2026-11-25 | **Lead call (planning lead)**, bite 5: still images are copied byte for byte into `frames/` as `<stem>_<path code>_0.<jpg|png>` (originals only read), counted apart from the extract plan, and skip clips, extraction and the filters; they join at frame review. | Spec §Stage details 1 (stills join at frame review); SPEC constraints 4 and 5. Review's `D` moves files, so working on the originals in place would move a source. Lead's reply to the builder, 2026-11-25. |
| D22 | 2026-11-25 | **Rule 1 amended by D31 (reviewed masks win).** **Lead call** (planning lead, SPEC Q2): overlap rules, applied when results are shown or exported; `sam_raw/` stays raw. (1) Two object masks with IoU > 0.5, from any prompts or classes: keep the higher score and flag it for review (spec §5). (2) Several prompts of one area class: union. (3) Two area classes on one pixel: the one listed first in `classes.yaml` wins; the starting file lists kelp before rock/substrate. (4) Objects sit on top of areas (spec); two objects that overlap less than that: the higher score takes the shared pixels. (5) A kelp shoot's pixels count as kelp in the class map and cover (`seg_as: kelp`); the shoot stays a counted object for the detector. (6) Pixels no class claims = unknown, id 0. (7) An untagged urchin shows as "urchin (untagged)" and is never exported (D18). | The spec gives rules 1, 4 and 6; the rest are gaps. Kelp on top of rock: kelp grows on rock, and kelp cover is the number the packet reports. Kelp shoot as kelp: the segmenter measures cover, the detector counts shoots, so shoot pixels would otherwise be missing from kelp cover. All of it is a settings-file field or one function; Nick can reverse it. |
| D23 | 2026-11-25 | **Lead call** (planning lead): tiling for frames wider than 2000 px: 1008 px tiles, 20 % overlap (spec, D10); SAM's image encoder runs once per tile and every prompt reuses it. Masks move to frame coordinates. Tile-edge duplicates: a mask that touches an inner tile edge is dropped when a mask of the same prompt from a neighbouring tile covers at least half of it; then masks of one prompt with IoU > 0.5 keep the higher score. The builder may change the rule in bite 2's plan if the preview shows it fails; QE checks. | Spec §5 says duplicates across tile edges are removed but not how. S1 counted urchins without removing them. A cut-off half urchin overlaps its whole twin by far less than IoU 0.5, so IoU alone would keep both. |
| D24 | 2026-11-25 | **Lead call** (planning lead, SPEC Q10, SAM part): confidence 0.4 (spec); inference resolution = full resolution in 1008 px tiles, frames up to 2000 px wide whole (SAM scales them to 1008 px). Minimum mask area: no guess; bite 3 measures the areas of real and false masks on the preview frames and the builder proposes a value in its PR, which the lead logs here. Memory guard: S4 (spec puts it under job control); S3 records peak process and Apple GPU memory per preview run so S4 can set it. | Measure, don't guess (CLAUDE.md value 3). D10 rests on n = 3 frames from one video: bite 3 re-checks it on at least 6 frames from at least 2 videos (urchins kept and seconds per frame, tiles vs whole frame). |
| D25 | 2026-11-25 | **Replaced by D30 (class form).** **Lead call** (planning lead): the Labeling page shows `classes.yaml` as a table and edits it in the browser (text box, Save, checked before an atomic write). The file is read with `yaml.safe_load` and written with `yaml.safe_dump` only (QE PR 19). Editing the file in a text editor and pressing Reload also works. PyYAML becomes a direct dependency once its licence is read from PyPI metadata (it is already installed through `sam3`). | SPEC constraint 7: every stage has a browser UI, and the terminal is only for starting the app. |
| D26 | 2026-11-25 | **Lead call** (planning lead): preview frames = 3 kept frames spread over the kept list (first, middle, last by name); "Other frames" picks 3 at random. Preview and Run each start one SAM worker process through `app/jobs.py`; the page shows the time per frame from the preview and an estimate for Run (seconds per frame × frames). | Spec §5 asks for 3 sample frames without saying which. A real estimate with warm-up and rolling average is S4 (spec, job control). |
| D27 | 2026-11-25 | **Lead call** (planning lead, QE PR 16 watch): still images stay byte-for-byte copies (D21); every place that reads pixels applies the EXIF orientation, and `sam_raw/` records the upright size. That includes S2's review-grid thumbnails (`app/stages/review.py`, which today make a sideways copy of a rotated still: QE PR 19, reproduced) and the blur and hash scores; S5's `export/images/` must hold upright pixels, so COCO boxes and `seg_masks/` sizes match the files. | Turning stills upright at ingest would break D21's byte-for-byte copy; one helper keeps SAM, masks and pages on the same image. |
| D28 | 2026-11-25 | **Lead call** (planning lead, D24): `min_mask_area_px` = 500 in the default `classes.yaml`. | S3 bite 3 bench (PR 22, DESIGN §Bench results): of 580 tiled urchin masks the smallest was 948 px; the 113 under 2500 px, cropped one by one, were real distant urchins except at most 2 unclear ones, and no small false masks appeared. 500 px only removes specks; a larger value would drop real far urchins. Nick can reverse it. |
| D29 | 2026-11-25 | **Lead call** (planning lead, D11): default prompts: kelp shoot (object) "young kelp", kelp (area) "kelp". | S3 bite 3 bench on 6 restoration-site frames from 2 videos (PR 22): "kelp stipe" found nothing; "young kelp" marked blades of real plants with no false masks; area "kelp" was tight but missed one dense dark patch; "kelp forest" covered open water. No prompt gives one mask per plant, so shoot counts need review in S5. n = 6 frames: re-check on more restoration footage. Nick can reverse it. |
| D30 | 2026-11-25 | Classes are added and edited through a form on the Labeling page (S3 bite 6, SPEC Q17); this replaces D25's text box. The form still writes `classes.yaml` (checked, atomic write, `yaml.safe_dump`), so the file stays the record and can still be edited by hand. | Nick, S3 builder chat 2026-11-25: editing YAML in a text box is too fragile; he wants a form like Roboflow's or Edge Impulse's. |
| D31 | 2026-11-25 | **Lead call** (planning lead, amends D22 rule 1): when a reviewed mask and an unreviewed mask overlap with IoU > 0.5, the reviewed one wins whatever the scores, and the pair is flagged for review; the unreviewed mask stays in `sam_raw/` but is not shown, counted or exported. Between two unreviewed masks, or two reviewed ones, the higher score still wins. | Spec §5: masks Nick already reviewed are never silently replaced. Under D22 rule 1 alone, a re-run's fresh duplicate with a higher score would hide a reviewed mask (QE PR 24). It matters from S5, when masks get reviewed; bite 4's `overlap.py` applies it now. Nick can reverse it. |

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

**S3 bite 3, 2026-11-25 — prompt preview bench.** SAM 3, confidence 0.4, Apple GPU, 1008 px tiles with
20 % overlap above 2000 px. Bench script and data: the builder's scratch folder on the SSD (not in git).

*D10 re-check, prompt "urchin", 9 frames from 5 videos (3 sites).* Frames up to 2000 px are never tiled, so
"tiles" and "whole" are the same run there.

| Frames | Size | Urchins kept, tiles vs whole | s / frame, tiles vs whole |
|---|---|---|---|
| Mendocino Headlands 2025, 2 frames, dense urchin barren | 3840×2160 | 160 vs 43; 210 vs 47 | 28–29 vs 3.6–6.8 |
| South Wreck 2024, 4 frames from 2 videos, close urchins | 5312×2988 | 23 vs 21; 52 vs 44; 17 vs 19; 42 vs 42 | 49–51 vs 6.7–17 |
| South Wreck 2021, 3 frames from 3 videos | 1280×960 | 7; 28; 41 (not tiled) | 1.8–4.0 |

Model load 13–15 s. Peak Apple GPU memory: 16.3 GB tiled, **26.7 GB whole-frame** (on the 5312 px frames;
S4's memory guard should know). Visual check, densest Mendocino frame with tiles: every mask sits on an urchin;
no masks on water, tape or bare rock; urchins on the far, hazy rock are missed. **D10 holds:** tiles keep 3–4×
more urchins where they are small and many, and the same number where they are close; the cost is 4–8× the time.

*D24 minimum mask area.* All 580 tiled urchin masks: smallest 948 px, median per frame 2.0k–30k px. The 113
masks under 2500 px, cropped one by one: all but at most 2 unclear ones are real distant urchins. No small
false masks appeared, so the minimum is only a floor for specks: **500 px** (below the smallest real mask).
A larger minimum would drop real far urchins.

*Kelp, D11: 6 restoration-site frames from 2 videos (Caspar South 2025, Fullers 2022), masks per frame.*

| Prompt (kind) | Masks per frame | Seen on the overlays |
|---|---|---|
| "kelp stipe" (object) | 0 / 0 / 0 / 0 / 0 / 0 | nothing |
| "kelp stalk" (object) | 0 / 0 / 0 / 1 / 0 / 0 | one plant's line |
| "young kelp" (object) | 0 / 0 / 0 / 3 / 6 / 0 | blades of the plants, in pieces; no false masks |
| "kelp" (object) | 0 / 2 / 0 / 3 / 14 / 0 | same plants, more pieces and overlaps |
| "kelp" (area) | 0 / 1 / 0 / 1 / 1 / 0 | the kelp plants, tight; misses one dense dark kelp patch |
| "kelp forest" (area) | 0 / 1 / 1 / 1 / 0 / 0 | one mask covers a large block of open water |
| "seaweed" (area) | 0 / 0 / 0 / 1 / 1 / 0 | same plants as "kelp" |
| "algae" (area) | 0 everywhere | nothing |

Two of the 6 frames have no kelp (a dive slate, a sea lion); zero is right there. Defaults: kelp shoot
"young kelp", kelp (area) "kelp". No prompt gives one mask per plant, so shoot counts will need review (S5).
n = 6 frames; re-check on more restoration footage when Nick has it.
