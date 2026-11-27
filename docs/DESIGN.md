# DESIGN.md — Architecture & Decisions (as-built)

*What it did / how it's shaped. Agents append; never silently rewrite history.*
*Last updated: 2026-11-27*

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
  **Replaced in S4 bite 1b:** SAM runs as a job of the job module; its job file and log are in `<data_dir>/jobs/`.
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

### Planned for S4 (job control) — lead's plan 2026-11-25 (CLAUDE.md rule 7); becomes as-built as bites land

```
app/jobctl/__init__.py   the module's whole interface: submit, status, list, pause, resume, cancel,
                         start_now, settings; kinds are registered by the app, not imported
app/jobctl/store.py      <data_dir>/jobs/jobs.sqlite: one row per job; settings.json next to it
app/jobctl/runner.py     the one queue thread: starts a job's worker process with the limits, reads its
                         "@@ {json}" lines, writes the log, sends "pause" on stdin, memory guard, schedule
app/jobctl/worker.py     helpers for worker processes: emit(), the stdin "pause" listener, the pause
                         between items, the thread cap; imported by app/sam/worker.py and the extract worker
app/jobctl/awake.py      keep-awake process on / off; battery check (pmset -g batt)
app/stages/*             register their kinds: build the worker's command from params, count items done /
                         total from their own files (sam_raw/, frames/)
app/web/templates/jobs.html   the Jobs page: list, Pause / Resume / Cancel / Start now, settings, summaries
```

- **Standalone (D4, D32).** `app/jobctl` imports nothing from `app.stages`, `app.sam` or `app.web`; a test
  checks the imports. Nothing in it knows what a frame or an urchin is: it sees items, counts and messages.
- **One queue, one job at a time** for the whole app (two SAM runs on the GPU triple the time, S1). Job records
  in `<data_dir>/jobs/jobs.sqlite`: id, kind, project, params (JSON), state (`scheduled`, `queued`, `running`,
  `pausing`, `paused`, `interrupted`, `done`, `cancelled`, `error`), items done / total, start_at, stop_by,
  keep_awake, limits used, created / started / ended times, stop reason, summary (JSON). The app never creates
  `data_dir` (D13); `jobs/` inside it is created on first use.
- **Worker protocol (D33).** Bite 1a builds the module with a fake job kind; bite 1b moves SAM onto it (its
  `frame` event becomes `item`). A job is a process: `python -m <module> <job.json>`. It writes `@@ {json}` lines:
  `loaded` (seconds), `item` (i, n, name, seconds, counts {label: n}, warning), `done`; anything else goes to the
  job's log. It reads stdin: a line `pause` (or end of file, when the app quits) means finish the item in
  progress, emit `paused` and exit 0. The worker skips items whose output is already on disk, so **progress is
  truth from the artifacts**: Resume starts a new worker for the same params and nothing is repeated or skipped.
  `jobctl` keeps its counters for display only.
- **Pause / Resume / Cancel (D34).** Pause sends `pause`; the process ends, so the model is unloaded. Resume
  queues the job again. Cancel = pause plus state `cancelled`; finished items stay. If the worker does not exit
  within 10 min of `pause` the app ends it; at most that item is lost. On app start, a job found `running` or
  `pausing` becomes `interrupted`; the dashboard shows "Unfinished job: Resume / Cancel".
- **Estimate (D35)** is a SAM job with params `{estimate: true}`: 1 warm-up frame not counted, then 5 frames,
  picked spread over the frames Run would label; results saved like any run. The Labeling page shows s / frame,
  frames left, total and finish clock time; during any job the remaining time is the rolling average of the
  last 20 items × items left.
- **Limits (D36)** in `<data_dir>/jobs/settings.json`, edited on the Jobs page, copied into each job at start.
  Device auto = NVIDIA GPU, else Apple GPU, else CPU (asked in a child process; the app never imports torch).
  Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `torch.set_num_threads` in the worker (default half of
  `os.cpu_count()`, 5 on this laptop). Priority: the worker starts with `nice` 10. Pause between items: the
  worker sleeps between items (still listening for `pause`). Memory guard: the runner reads free memory every
  2 s (one named figure, picked and measured in bite 4: D36); below the threshold it pauses the job with the
  reason "memory guard: N GB free".
- **Schedule (D37).** Start at = the next time that clock time comes; Stop by = the next time after the start.
  The runner checks every 15 s. A scheduled job survives a restart (it is a row); if the app was closed at its
  start time and Stop by has not passed, it starts when the app starts; otherwise it shows "missed". The page
  says the app must stay running for a schedule to fire.
- **Keep awake (D38).** The app starts one macOS keep-awake process (`caffeinate -i -s -w <app pid>`) while any
  keep-awake job is scheduled, queued or running, and ends that process (by the PID it started, never any other
  one) when none is left; `-w` also ends it if the app dies. Reminder next to the toggle: plugged in, lid open.
  Reading `pmset -g batt` shows a warning on battery. The app never changes power settings. Tests use a fake
  command; builders do not run the keep-awake command or change power settings from their own shell (a hook on
  this laptop blocks it); the self-test goes through the app's toggle.
- **Log and summary.** `<data_dir>/jobs/<id>.log`: start, limits, each item, warnings, pauses with reason,
  errors, end. Summary (written at the end and on each pause): items done, sum of `counts` per label (for SAM:
  masks per run unit, so urchins found), errors, time taken, stop reason. The dashboard shows the last finished
  job's summary.
- **As built, bite 1a (PR 32).** `JobControl(<data_dir>/jobs)` with `register(Kind)`, `submit`, `status`, `list`,
  `pause`, `resume`, `cancel`; a `Kind` is a worker module, `prepare(params)` (the job file, `jobs/<id>.json`,
  rewritten on each start) and an optional `progress(params)` counted from the kind's own files. State changes are
  compare-and-set in one SQL statement. Workers start in their own session, so a Ctrl-C on the app does not cut an
  item in half: the worker reads end of file, finishes the item and exits; `emit()` exits quietly if the app is gone.
  QE PR 32: the worker's PID is kept in the row (`worker_pid`) until the runner has waited for it; no job starts
  while a recorded worker is still alive (its command line must name `jobs/<id>.json`, so a reused PID does not
  count), and one that is still alive 10 min (D34) after the queue began waiting for it is ended. An error while
  following a worker ends that worker before the queue moves on. `jobs/app.lock` (`flock`) allows one app per data
  folder; a second `uv run app` on it exits with a message.
- **As built, bite 1b (PR 33).** SAM Preview and Run are jobs of kind `sam` (params: folder, frame names, preview,
  predictor); `label.build_job` makes the worker's job file at each start or resume, and `label.progress` counts
  frames with nothing left to run in `sam_raw/`. The worker writes `sam_preview.json` itself and its `done` event
  carries the page's final message. S3's in-process driver (`label.run`, `run_all`, `run_preview`) and Stop are gone.
  Starting a new Preview or Run cancels the project's older paused or interrupted SAM job (the new one skips what
  it finished). The SAM job file and log are `<data_dir>/jobs/<id>.json` / `.log` like every kind, not
  `<project>/sam_logs/` (builder call: one place for every job's log; the Jobs page names it). On start, an
  interrupted job's counts are recounted from its artifacts. Pause / Resume / Cancel are `POST /jobs/<id>/<action>`
  with a same-app `back` path. A kind names its items (`Kind.item`: SAM says "frame 3 of 6").
  Known gap until bite 2: extraction still runs on `app/jobs.py`, so an extraction in one project and a SAM job in
  another can run at the same time; bite 2 puts extraction on the job module and closes it (PR 34).
- **As built, bite 2 (PR 34).** Extraction is a job of kind `extract` (`app/stages/extract_worker.py`; params: the
  plan as it was when Extract was pressed; the blur and duplicate values are read from `project.json` each time
  the worker starts, so an Apply made while the job is paused counts on Resume: QE PR 34); `extract.run` checks for a pause before each
  frame and skips frames on disk, as before. Playback copies are jobs of kind `playback`
  (`app/stages/playback_worker.py`, one per video); it cannot pause (`Kind.pausable = False`): Cancel kills the
  worker's whole process group, ffmpeg included, and its output is under temporary names (D15). Workers send
  `progress` (percent + message, for a kind without items) and `error` (the message the page shows, e.g. "Video not
  found: …") events. `app/jobs.py` is removed, so every background job in the app is on the one queue. Extract
  frames with new settings cancels a paused or interrupted extraction (the new plan replaces it). Tests reach the
  extract worker through `extract.GRAB_HOOK` (as SAM's `PREDICTOR`), since a test cannot patch another process.
- **As built, bite 3 (PR 35).** Estimate is a `sam` job with params `estimate: true` on 1 + 5 frames picked spread
  over the frames Run would label (`label.estimate_frames`); the worker writes `<project>/sam_estimate.json` (warm-up
  frame, the 5 timed frames, seconds per frame, and `results.classes_key` of the classes and settings). Run's plan
  uses it (seconds per frame × frames with work) while that key matches the current classes, else the preview-based
  guess (D26), and says which; a changed class shows the estimate as out of date. The finish time is "if started
  now" until the schedule (bite 5a) adds a start time. Remaining time of any job: the runner keeps the seconds of
  the last 20 timed items (skipped ones excluded) and stores `eta_s` = mean × items left on each item; the Labeling,
  Extract, Clips and Jobs pages show it.
- **As built, bite 4 (PR 36).** `app/jobctl/limits.py`: `<data_dir>/jobs/settings.json` (device, threads, low
  priority, pause between items, memory guard on/off and GB, batch 1), checked before saving from the Jobs page's
  form; a missing or unreadable file gives the defaults. At each start the runner copies them into the job's row
  and job file, starts the worker as `nice -n 10 python …` (nice execs, so the PID is the worker's) with
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS`, and starts a guard thread that reads `limits.free_gb()` every 2 s and
  pauses the job below the threshold with the reason "memory guard: N GB free, below T GB". The guard's figure is
  psutil's `virtual_memory().available` (free + inactive pages on macOS, psutil source); `vm_stat` "Pages free"
  reached 0 GB during ordinary runs, so it cannot be the figure. SAM: device "auto" = NVIDIA, else Apple GPU, else
  CPU (asked in a child process); `torch.set_num_threads`; a PyTorch "fall back to run on the CPU" warning becomes
  a `warning` event, kept once per op on the job and shown on the Jobs and Labeling pages. Workers rest
  `pause_s` between items through `worker.rest` (a pause wakes them).
- **As built, bite 5a (PR 37).** `app/jobctl/schedule.py`: "HH:MM" → the next time it comes (Stop by: the next
  time after the start, or after now without a start). `submit(..., start_at, stop_by)` stores a `scheduled` row;
  the queue thread checks every second (D37 said 15 s; the check is one query) and queues a job whose start has
  come, also after a restart, unless its Stop by has passed too: then it is `missed` (a new finished state). A
  running job gets a watcher that pauses it at Stop by ("stop by 7:35 PM"); a queued job whose Stop by passes while
  it waits becomes paused; Resume clears a Stop by that has passed and keeps one still to come (QE PR 37). `start_now` and Cancel work on a scheduled job; a project
  with a scheduled extraction or SAM job refuses another until it is started or cancelled. The Labeling page's
  finish time counts from the scheduled start.
- **As built, bite 5b (PR 39).** `app/jobctl/awake.py`: one keep-awake process (D38's command, with `-w` the app's
  PID) held by an `Awake` object; a JobControl thread checks every 2 s whether any keep-awake job is scheduled,
  queued, running or pausing, starts the process if so and ends it (its own Popen only) if not; submitting a
  keep-awake job starts it at once. A missing command is shown on the Jobs page, never raised. `on_battery()` reads
  `pmset -g batt` (read only). "Keep laptop awake" is a checkbox on Run and Extract with the plugged-in / lid-open
  reminder and a battery warning; the Jobs page says when the process is on. Summary: the runner keeps one per job
  across Pause / Resume (items done, counts per label summed from `item` events, errors, seconds, state, stop
  reason, warnings, the kind's item word), written to the row on each item and at each stop, and to the log. The
  dashboard shows the newest stopped job's summary (finished or paused, so a Stop by pause counts), urchins first.
  `GET /jobs/<id>/log` shows a job's log as text.
- **As built, bite 6 (Nick's S4 test-build feedback).** Cancel always ends the worker's process group at once
  (D34a): `JobControl.cancel` asks the runner with `kill=True` for every kind, also while the job is pausing; the
  stop reason is plain "cancelled". Every worker writes each item under a temporary name and renames it, so a
  killed item is never half-written. A pause request sets the message to "finishing <item> <n> of <N>,
  about <m> min before pausing" (mean of the timed items), and the Labeling page offers Cancel while running or
  pausing. Run panel:
  `run_plan` adds `per_frame`; `job_view` adds `line` (time left, finish clock time, then the message) used by
  the Labeling page, its status poll and the Jobs page. D36b: `HARMLESS_FALLBACKS` in `app/sam/worker.py`; such a
  fallback is a `note` event, which the runner writes to the log only.
- **As built, bite 6b (Nick's S4 test, items 4–8).** `templates/_job.html` holds three macros used by every
  page and by the live fragments: `job_buttons`, `job_panel` (state, buttons, progress bar, a CSS "running" sign,
  time left and "done by", a memory-guard explanation; `compact` for the project page) and `waiting_banner`
  (paused, pausing or interrupted jobs). Fragments: `GET /live/jobs` (the Jobs table), `GET /live/banner?project=`
  and `GET /p/<name>/live/<sam|extract>` (the latest job of that kind). `static/live.js` polls each `[data-live]`
  box every 2 s while the tab is visible and replaces it only when the fragment's HTML changed (never between a
  mouse down and its click); `data-reload-on="done error"` reloads a page once when its running job ends that way
  (Labeling: new results and Run plan; Extract: new frames); buttons with `data-job-lock` follow the job's state.
  D36c: the guard's reason starts with `GUARD_REASON` ("Paused by the memory guard: 1.8 GB free (limit 2 GB)");
  `JobControl.auto_resume()` runs in the 2 s keep-awake thread and resumes such a job once free memory has stayed at
  or above twice the job's own limit for `RECOVER_S` (60 s), logging it; the summary counts `memory_pauses` and
  `auto_resumes`. Pauses by hand, Stop by and Cancel never resume by themselves; a guard-paused job whose Stop by
  has passed stays paused and its reason becomes that Stop by (QE PR 42).
- **As built, bite 6c (PR 45; Nick's S4 test, items 9, 9a, 8a).** `label.labeled_frames(folder, files)` lists the
  kept frames with a `sam_raw` file, newest first, by a directory listing only. `GET /p/<name>/labeling/labeled?n=&since=`
  returns the newest n with counts per run unit and a `sig` (count + newest mtime + n); when `since` equals it the
  reply is just `{"sig", "same"}`, so the page polls it every 3 s cheaply. `labeling.js` keeps a list of grids
  (preview, "Labeled so far") sharing one legend state and the large viewer; the "Labeled so far" tiles are built
  in JS and carry `v=<mtime>` on the overlay URL so a relabeled frame is not served from the browser cache; the
  grid is not rebuilt while the large viewer shows it. The viewer and `#frame-list` are on the page without a
  preview too. 48 tiles at a time ("Show 48 more", `?n=`). Project page: the SAM labeling row's state and counts
  come from the results on disk ("done" = every kept frame has a results file; a class added later shows in the
  Run plan, not here). UI spelling "Labeled" (American English, team rules §7).
- **Job kinds (app side).** `sam` (preview, estimate, run; worker `app.sam.worker`, items = frames, counts per
  run unit), `extract` (worker new in bite 2, items = planned frames), `playback` (copy + strip per video; no
  pause needed). Every kind's job file and log are `<data_dir>/jobs/<id>.json` / `.log` (bite 1b).

### Planned for S5 (review labels and export) — lead's plan 2026-11-25 (CLAUDE.md rule 7); becomes as-built as bites land

```
app/review/state.py      review tables in state.sqlite; a frame's final masks = sam_raw + review, overlap rules
app/review/tagging.py    urchin types, "tag all remaining", next untagged
app/export/writers.py    images, COCO, seg_masks + classes.json, frame_metrics.csv
app/export/splits.py     site and dive date from names; group split (D43)
app/export/checks.py     COCO load, mask sizes and ids, metrics vs masks
app/stages/review_labels.py   page glue for the Review page; app/web/templates/review_labels.html, static/labelreview.js
app/web/templates/export.html  the Export page
```

- **Review state (D40).** `sam_raw/` stays what SAM returned plus a stable `mask_id` per mask; masks added by
  hand go into `sam_raw/<frame>.json` under the unit `added` with `added_by_hand: true`. Everything a reviewer
  does lives in `state.sqlite`: `mask_review` (mask_id, frame, class_id, deleted, reviewed, time; the urchin type is the class id, a class of
  group `urchin`, as-built bite 1)
  and `frame_review` (frame, accepted, split_ok, checked classes, time), plus an action log for undo. A re-run
  keeps reviewed masks (S3, SPEC Q7) with their ids. Export is rebuilt from `sam_raw` + review state, so
  re-exporting is always safe (spec).
- **Accepted frames (D41).** Only accepted frames are exported. A frame cannot be accepted with an untagged
  urchin (D18). Frames with split check "no" count for cover (`seg_masks`, metrics) but are left out of
  `detector_coco.json` (spec). Classes not marked fully checked on a frame are listed per image in COCO
  (`checked_classes`) and in `frame_metrics.csv`, so training can skip them (spec).
- **Metrics (D41, SPEC Q11).** Counts and % cover come from the final class map (D22, after review). Largest
  object % = pixel area of the largest object mask / frame area. Unknown = id 0 (D22).
- **CVAT (D42, SPEC Q9).** Fine outline fixes use CVAT on its own: `detector_coco.json` carries each outline as a
  polygon (pycocotools), which CVAT's COCO importer reads (the builder cites CVAT's docs for the format before
  relying on it). Fixes are not imported back in Build 1 (icebox).
- **Splits (D43).** Site and dive date come from the file name (`<site>-<date>-<time>-<diver>-<tag>`, SPEC facts;
  the date's exact format is read from the real names in bite 2b and written down there) or else the folder; one group per site + date; groups go to train / valid / test to come closest to 70 / 20 / 10 by
  frame count, same answer every time. With fewer than 3 groups a fair split is impossible: the Export page
  says so and splits by video instead.
- **Export images (D44).** Upright pixels (D27): files without an EXIF rotation are copied byte for byte, others
  are re-saved upright at JPEG quality 95. `seg_masks/` are 8-bit PNGs, pixel = class id; 255 (untagged) never
  appears, since such frames cannot be accepted.
- **Area seams (D45).** Bite 7 compares area classes on the whole frame against tiles and the lead logs the pick.
- **Add a missed object (D46).** A click or box runs SAM 3's point / box prompt on that frame through the job
  module. If one add takes more than 20 s (model load ~11–17 s, S4 logs), the builder keeps the worker loaded
  while the Review page is open (ends after 10 min idle) and says so in its PR.
- Review state (as-built, bite 1): `mask_id` is an integer unique in its frame file, never reused
  (`next_mask_id`); files from before S5 get ids in file order when read (reading never writes). `app/review/state.py`
  lays `mask_review` over `sam_raw` (`overlay`) and applies the overlap rules (`final`, `class_map`). A mask is
  reviewed after any action on it or when its frame is accepted; the SAM worker passes the reviewed ids to
  `results.write_frame`, which keeps them. A deleted mask is never shown but still wins rule 1 against unreviewed
  masks of its **own run unit**, so a re-run does not bring it back; it never hides another class's detection
  (QE PR 44). `split_ok` is NULL until marked; only "no" leaves a frame out of COCO. Every action logs the rows it
  changes as they were, and undo restores them (a stack per frame or project).
- Export (as-built, bites 2a + 2b, `app/export/`): `writers.run` builds everything into `export.new/`, runs
  `checks.run_all` on it, and only when every check passes renames `export/` → `export.old/`, `export.new/` →
  `export/` and deletes `export.old/`; a run that finds `export.old/` without `export/` (a crash between the two
  renames) puts it back first. A failed check leaves the new build in `export.failed/` and `export/` unchanged; a
  build error (e.g. an accepted frame that a re-run gave an untagged urchin) raises and keeps the old export.
  COCO outlines: masks → polygons with `contourpy` (marching squares at 0.5; BSD-3-Clause, PyPI metadata of
  1.3.3; `<1.4` because 1.4 needs numpy 2; pycocotools only goes polygon → RLE), Douglas-Peucker at 0.5 px, holes
  dropped (COCO polygons are filled); on 109 real masks IoU vs the mask 0.964–0.997. Category id = class id.
  Metrics counts are the final objects after review and rule 1; cover from the class map. macOS writes a `._<name>`
  file next to every file on the exFAT SSD (`com.apple.provenance`); the checks ignore `._*`.
  D43 date format: `<site>-<yy>-<mm>-<dd>-…` read as year-month-day (QE PR 46 on Nick's footage: 30 of 46 videos
  match, fields 22/24/25, 06–10, 11–29, so not dd-mm-yy; 6 site + date groups); the others use the folder name.
- Review page (as-built, bite 3a): `/p/<name>/labels` (`app/stages/review_labels.py`, `review_labels.html`,
  `static/labelreview.js`), linked from the project page's "Review labels + export" row. It lists the kept frames
  that have results (accepted / to do) and opens them in S2's `viewer.js`, reused with two small hooks: a page
  may take a click on the picture (`window.viewerPick`, image pixels; label review selects the mask there instead
  of zooming), and `viewerZoom(factor)` for the zoom slider; keys typed into a box no longer move frames. The
  server draws the picture (3200 px, never up) from the frame's final masks after review, in four modes (all,
  objects only, areas only, final class map), with the selected mask outlined in white; `at?x=&y=` returns the
  mask under a pixel (objects first, smallest first, then areas). Actions (`act`: class, delete, accept, reopen,
  undo) answer the frame's new view or 400 with the reason, e.g. accept with untagged urchins. The page keeps
  no review state of its own.
- Urchin types (as-built, bite 4a): black urchin is id 10 in group `urchin`. S3's "an id with results keeps its
  name" check is lifted (D39); "a name with results keeps its id" stays, except when that id is in the file under
  another name. A run unit's key is its name (group name or class name) and `sam_raw` stores results by that key,
  so every save of `classes.yaml` calls `results.follow_renames`: it works out each saved unit's new key from its
  class ids first (so names may swap or rotate in one save, QE PR 47), never merges two saved units into one, and
  updates the names and `class_ids.json`; nothing re-runs. Builder choice: a type's key is the first letter of its
  name (P, R, B; a clash or X / Z / F gives none), so a renamed type gets its new letter without a new file field.
- Split check, fully checked, totals (as-built, bite 3b): the review panel has "touching urchins separated" yes /
  no / not marked and one "fully checked" box per class (all classes, D41). Running totals = objects per object
  class on the **accepted** frames that go to the detector (split check not "no") and the number of such frames,
  against the 150–200 target, plus the objects on accepted split-"no" frames apart (cover only, QE PR 51); each
  frame's counts are cached in the app process until its `sam_raw` file, its review rows or the class list change.
- Add a missed object (as-built, bite 5a): a job of the kind `sam_add` (not pausable; one click or box per job,
  so it queues behind a Run and never shares the GPU, D32). The worker loads SAM 3 with its click / box head
  (always SAM 3: SAM 3.1 has none of those weights, D10), looks at a square crop around the spot (1008 px, or twice
  the box), and saves the mask under the unit `added` with `added_by_hand`, `reviewed: true` (a re-run never
  replaces it; it wins rule 1 against unreviewed masks, D31) and the job's request id (a resumed job never adds
  it twice). Real SAM on a 4K frame: model load 14.1 s + click 2.0 s = **16.9 s per add**, under D46's 20 s, so
  no worker is kept loaded; the clicked urchin's mask matched SAM's own text mask of it at IoU 0.87.
- Fast tagging (as-built, bite 4b): the frame view lists the urchin types with their keys, which masks are urchins,
  the objects in reading order and each object's zoom rectangle (`tagging.focus`). `Tab` selects the next untagged
  urchin and zooms the viewer onto it (`viewerFocus`, a third opt-in hook in `viewer.js`); a type key (P / R / B,
  the first letter of each type's name) or a type button tags the selected urchin **and moves on to the next
  untagged one** (builder choice: one key per urchin); "Tag all remaining as <first type>" tags the rest in one
  undo step. A close-up panel shows the frame at full resolution around the selected mask (480 px, outline only,
  so its colour is easy to judge). The panel toggle is the backtick key, so letters stay free for types (QE).
- Export page (as-built, bite 6a): `/p/<name>/export`, linked from the project page and the Review page. Export
  is a job of the kind `export` (`app/export/worker.py`, not pausable: the export is rebuilt from scratch anyway),
  one item per accepted frame; refused with no accepted frame or while one is queued or running. `writers.run` also
  writes `summary.json` (counts, splits, the D43 warning, `ok`, time) next to `checks.json`; the page shows the
  newer of `export/` and `export.failed/`, so a failed export is never hidden behind an older good one, and says
  when `export/` was not replaced. The page reloads itself when the job ends (`live.js` `data-reload-on`).
- Add a missed object on the page (as-built, bite 5b): "Add a missed object" + a class drop-down (object classes)
  turns on add mode for one add: a click, or a dragged box (`viewer.js` box mode: `viewerBoxMode` / `viewerBox`, a
  fourth opt-in hook; without it a drag still pans), goes to `POST /labels/add/<frame>` in frame pixels; the server
  checks the class and the request and queues a `sam_add` job. The page polls `/labels/job/<id>`, then reloads
  the frame and selects the new mask. Esc or the button cancels add mode without closing the viewer.
- Licences, README, CVAT (as-built, bite 6b): `THIRD_PARTY_LICENSES` carries the SAM License word for word from
  the `LICENSE` file of `facebook/sam3` in the Hugging Face cache (byte-identical to `facebook/sam3.1`'s, `cmp`
  2026-11-26) and contourpy's BSD-3-Clause text. D42's CVAT route, checked against CVAT's COCO format page
  (docs.cvat.ai/docs/dataset_management/formats/format-coco/, read 2026-11-26): the COCO 1.0 importer lists
  polygons and masks as supported; its documented route is to create a task with the images and then
  "Upload annotation" → "COCO 1.0" (QE PR 57: the page shows it while creating a task from a COCO dataset). The page does not say how `images[].file_name` is matched or whether extra keys (`checked_classes`,
  `score`, `split`, `seg_mask`, `added_by_hand`) are ignored on import: **not tested end to end** (no CVAT on the
  laptop); the first real CVAT import is the test.
- Tests: synthetic frames and a fake predictor (no torch, weights or real frames); the builder runs real SAM in
  the browser for bites 5 and 7.

### Planned for S6 (what runs where, on the laptop) — S6 builder's bite 1, 2026-11-27; becomes as-built as bites land

**What is reused** from `ADIN_SPI_OpenMV/ml/` (Nick's earlier work; read 2026-11-27; nothing is written there):

| Earlier piece | Reused as | How it is brought in |
|---|---|---|
| `yolox_urchin/model.py` `build_model` (conv stem) and `RawExport` (three raw head maps, sigmoids baked in); the same code in the gate's `export_candidates.py` | the YOLOX candidate's model definition (bite 2a) | adapted into `app/modeldefs/`; Megvii's YOLOX stays an unmodified checkout at `6ddff48` (D51) |
| `yolox_urchin/export.py`: ONNX (opset 13, classic exporter) → `onnx2tf -oiqt` with a calibration file → `*_full_integer_quant.tflite`; pre-seeding onnx2tf's sample-data file | the 8-bit step of the `convert` job (bite 3a) | adapted into `app/fit/`; the DUO calibration frames are not reused (D55) |
| `compile_model.sh`: the two compilers' arguments, run through OpenMV's `tools/modelc.py` in the `firmware-builder` container | the N6 compiler step (bite 3b), with the SDK, the OpenMV tree and the input mounted read-only | the command is rebuilt in `app/fit/`; the script itself is not called |
| the Vela arguments in `compile_model.sh` and OpenMV's `tools/vela.ini` | the AE3 compiler step (bite 3b), run natively (D52) | same arguments; `vela.ini` read from the OpenMV tree |
| `compile_gate_report.md`'s method: untrained model, placement from the compilers' own reports (Vela's summary CSV and operator counts; ST's `network_generate_report.txt` "Epochs details") | the fit report's fields and parsers (bites 3a, 3b) | the two real reports of the known-good run become parser test fixtures (text only, no weights) |
| `build_romfs_n6.sh` (one model image holding ours and the vendor's models) | S7's load step and bite 4's "stored together" check | read only in S6: the 24 MiB model area (`boards/OPENMV_N6/board_config.h`, `OMV_ROMFS_PART0_LENGTH 0x01800000`) is the limit a file is checked against |

**Flow of one cell** (candidate × camera): build the model untrained in the conversion environment → ONNX →
8-bit TFLite → the camera's compiler → `fit.json`. The job worker runs in the app's environment (no torch needed)
and starts each step as a child process in the tool environment it needs (D51). Each step's output on disk is
its progress (D33): a step whose output exists is skipped on Resume.

**Folders** (D53):

```
<tools_dir>/                         on the laptop's own disk (D51); never in git
  convert-venv/                      Python 3.11; torch → ONNX → 8-bit TFLite; LiteRT
  vela-5.0.0-venv/                   Arm Vela (AE3)
  third_party/<name>/                model code checked out at a pinned commit (YOLOX 6ddff48, ...)
  licences/                          licence texts read before use (ST's); also copied to <data_dir>/fit/licences/
<data_dir>/fit/<arch key>/           untrained candidates: app-wide, no project
<data_dir>/projects/<name>/models/<model id>/    trained models (S8)
  recipe.json                        D54
  checkpoints/  train.jsonl          S8 adds these
  int8/                              model.onnx, model_int8.tflite, int8.json (how it was made), convert.log
  targets/<n6|ae3|pi>/               the compiler's output file, compile.log, the compiler's own report, fit.json
```

`fit.json` (one per model × camera; the "What runs where" page reads only these): `converts` yes / no;
`placement` {npu, hybrid, cpu, total, unit: "epochs" (N6) or "operators" (AE3)}; `failing_layers` [{name, type,
why}]; `file_bytes` and `limit_bytes` with the limit's source; `memory_bytes`; `estimate_ms` marked as the
compiler's estimate; `tool` name and version; `error` (the compiler's message when it does not convert). A failed
conversion writes `fit.json` with `converts: false`; the job itself ends "done".
### Planned for S8 (dataset versions and the first detector) — builder's plan 2026-11-27 (TRACKER S8 bite 1); becomes as-built as bites land

```
app/datasets/versions.py   freeze a version from export/ (manifest, hash), list and read versions
app/datasets/splits.py     carry each group's split from the version before; deal new groups (D61)
app/datasets/view.py       what training will see: counts, classes under target, frames left out (bite 2b)
app/datasets/loaders.py    detector samples from a version, honoring "fully checked" and the split check (bite 2b)
app/datasets/outside.py    public sources from the outside-data folder, with their licence records (bite 3)
app/datasets/worker.py     the `freeze` job kind
app/train/worker.py        the `train` job kind: epochs as items, checkpoint + metrics line per epoch (bite 4a)
app/train/yolox.py         the YOLOX trainer behind it (bite 4b; needs S6 bite 1's decisions)
app/web/templates/datasets.html, train.html
```

- **Layout of `datasets/` (D60).** Per project:
  ```
  datasets/images/<sha256>.<jpg|png>   every frame any version uses, stored once under its content hash
  datasets/v<N>/manifest.json          see below; written once, never changed
  datasets/v<N>/detector_coco.json     the export's file with each image's `split` set to the version's
  datasets/v<N>/frame_metrics.csv      the export's file with the `split` column set to the version's
  datasets/v<N>/seg_masks/             copied from export/ (8-bit PNGs + classes.json)
  datasets/v<N>/splits/                train.txt, valid.txt, test.txt (the version's, not the export's)
  ```
  Labels are frozen by copy (they are small). The export's own `split` fields would be wrong from version 2
  on (D61), so the version's copies carry the version's splits and the manifest is the authority.
  "Frames by reference" (spec §Outputs) = the manifest names each frame by its content hash and the file
  lives once in `datasets/images/`, shared by every version that uses it. A reference into `export/images/`
  or `frames/` would not hold: the export is rebuilt from scratch on every Export (D41), a frame rejected
  later moves to `rejected/`, and rotated stills are re-saved (D44). The SSD is exFAT (as-built S5), so hard
  links are not available. Cost: one extra copy of each accepted frame per project, not per version.
  A version is built in `datasets/v<N>.new/` and renamed when complete; a half-built one is removed on the
  next freeze. Nothing in the app rewrites or removes a `v<N>/` folder or a file in `datasets/images/`.
- **`manifest.json`.** `number`, `name` (Nick's words, optional), `created`, `hash`; `export` (the export's
  `summary.json` time and counts); `classes` (the export's `classes.json` list); `files` (sha256 of each
  copied label file); `frames`: one row per frame with `file`, `image` (sha256 + extension), `width`,
  `height`, `source`, `group`, `split`, `split_ok`, `checked_classes`; `groups` (`group`, `split`, `frames`,
  `carried` from which version or `new`); `grouped_by` and the export's split warning; `counts` (objects and
  frames per class and split); `outside` (bite 3: sources with licence, counts and their own content hash);
  `previous` (the version it carried splits from). `hash` = sha256 of the canonical JSON of `classes`,
  `files`, `frames`, `groups` and `outside`: the same content gives the same hash whatever its number, name
  or date, and Freeze says "nothing changed since v<N>" instead of writing an identical version.
- **Freeze is a job** (kind `freeze`, not pausable: it copies files into `v<N>.new/` and starts over if
  stopped). Refused with no export, with an export whose checks did not pass (`summary.json` `ok`), or while
  an export job is queued or running.
- **Splits across versions (D61).** Version 1 takes the export's groups and splits as they are. Later
  versions never ask the export where a frame goes: a frame that was in any earlier version keeps the split
  it had in the newest version that holds it (so a frame left out of one version and back in the next is not
  dealt again); a group (site + dive date, or video when the export had to fall back, D43) that was in an
  earlier version keeps its split the same way, and its new frames join it. "The same frame" is the same
  file name, not the same image hash: a rotated still is re-encoded on every Export (D44), so its bytes can
  change while it is still that frame (QE PR 61). New groups are dealt, largest first, to the split furthest
  below 70 / 20 / 10 counting the frames already placed (the export's rule, started from the carried
  counts). One case the spec does not cover: the export's grouping changes from "video" to "site and dive
  date" once a project has 3 dives, and a dive whose videos sat in different splits becomes one group. Its
  old frames keep their splits (a test frame never moves to train); its new frames go to the most held-out
  split its old frames are in (test before valid before train), and the page lists such groups.
- **Outside data in a version (as-built, bite 3; QE PR 66).** `datasets/v<N>/outside/<key>.jsonl` holds one
  row per image of an added public source (path inside `outside_data_dir`, size, split, boxes, size in bytes);
  the manifest entry carries the licence, attribution, counts and the rows file's sha256, which is part of the
  version's hash. The images stay where they are and are frozen by their size in bytes only, not by content
  hash: weaker than D60 gives Nick's own frames, chosen because the folder is tens of GB and its archives are
  never modified; the entry says so (`frozen_by`). Sources and their records: `app/datasets/outside.py`,
  `THIRD_PARTY_LICENSES` "Outside data" (D63).
- **Training reads a version, never the export** (spec stage 7): the loaders take `datasets/v<N>/` and
  `datasets/images/` and check each image's hash on first use.
- **What training adds to S6 bite 1's decisions** (package location, `models/` layout, recipe: the S6
  builder's calls, now Lead calls D51, D53 and D54; nothing here changes them). S8 uses that layout's names as they
  are: a trained model is `projects/<name>/models/m<number>-<arch key>/` with `recipe.json`, `checkpoints/`
  and `train.jsonl`. What S8 puts in them: `recipe.json` → `train` holds the dataset version and its hash,
  classes, tiling, epochs, augmentation preset, seed and the starting weights with their licence record
  (D54), and the train job appends one `recorded` entry per start or resume (time, where it continued from,
  code commit, library versions, device, limits); `checkpoints/last.*` is where Resume continues (model,
  optimizer, EMA, epoch and step: the earlier trainer's dict plus the step) next to the best one by
  validation; `train.jsonl` has one line per finished epoch; `val_frames/` (new, S8) holds a few drawn
  validation frames per epoch for the Train page. Bite 4a builds the job kind with a fake trainer against
  exactly these files, so 4b only swaps the trainer in.

**Reuse from the earlier YOLOX work** (`ADIN_SPI_OpenMV` @ `994b8c2`, `ml/yolox_urchin/`, read 2026-11-27;
read only, nothing is imported from that repo: pieces are adapted into `app/train/` with a note in each file).

| Earlier file | What it is | Reused in S8 |
|---|---|---|
| `train.py` | the loop: SGD + Nesterov, peak lr 0.01 / 64 × batch, 3-epoch linear warm-up (at most 1500 steps), cosine decay to 5 %, EMA with ramped decay (fixed decay scored 0.478 against 0.573, its own note), BatchNorm eps 1e-3 / momentum 0.03, mosaic off for the last 10 epochs, checkpoint dict `{model, opt, ema, epoch}`, resume, run record (config, git sha, corpus hash) | **Adapted** (4b): same loop and numbers as the known-good start (value 5). Changed: SIGTERM stop → the job module's `pause` on stdin; `~/nereus_ml/runs` → the model's folder; one class → the run's class list; one `@@` metrics line and checkpoint per epoch |
| `model.py` | `build_model` (YOLOX-Nano / -Tiny from YOLOX's own `exp`, Focus stem swapped for a stride-2 conv, the variant that compiled for both NPUs), three run-time patches for the Apple GPU (legacy `.type()` casts), `RawExport`, `decode_raw` | **Shared with S6** bite 2a (the detector's model definition is the S6 builder's); the trainer calls that definition. The Apple GPU patches are training-only and move to `app/train/` |
| `data.py` | samples: scale so the median box lands in the 24–64 px band, box-aware crop to a 256 px canvas (15 % random), mosaic, flip, HSV, Gaussian blur on its own random stream so turning blur on never moves a box; boxes under 8 px dropped | **Adapted** (4b) as the recipe's tiling and augmentation. Uses OpenCV, which the app does not depend on: 4b first tries Pillow / numpy / scipy (already dependencies) and adds OpenCV only with its licence read. **New:** haze (TRACKER 4b asks for it; the earlier file left haze out on purpose: "measured harmless to tiny", hurt only Nano) |
| `test_data_aug.py` | boxes unchanged by blur, same seed → same sample, evaluation never blurs | **Adapted** as tests (4b), on synthetic images |
| `train_ctl.py`, the LaunchAgent in `README.md` | its own control page, SIGSTOP pause, night schedule | **Not reused**: `app/jobctl` does all of it (D4, D32–D38) |
| `export.py`, `decode_np.py`, `eval_rung_a.py` | 8-bit export and compiles; numpy decode + NMS; COCOeval scoring | Not S8: S6 (convert) and S9 (evaluate) |
| `train_species.py`, `stage2_autobox.py`, `harvest_crops.py`, `hil_stills.py`, `plot_curves.py` | the two-stage species head and rig tools | Not reused: Build 2 trains the urchin types as detector classes |
| `~/nereus_ml/datasets/corpus_v1/` and the runs' weights | Urchinbot + DUO + RF100 corpus; stage-1 weights trained on it | **Not used** (spec, Nick 2026-11-27: no DUO, no DUO-trained weights); bite 3 builds its own view of the allowed sources |

Licences for S8, read at the source 2026-11-27 (texts and details in `THIRD_PARTY_LICENSES`):
YOLOX code Apache-2.0 (`Megvii-BaseDetection/YOLOX` `LICENSE` at `6ddff48`, the head of `main`; GitHub's
licence API says Apache-2.0; the local checkout `~/nereus_ml/third_party/YOLOX` is the same commit and its
`LICENSE` has the same sha256). YOLOX's published COCO weights: the repository has no separate licence
statement for them; they are release files of that repository, trained on COCO, whose terms put the
annotations under CC BY 4.0 and leave the images under Flickr's terms (`cocodataset.github.io`
`dataset/termsofuse.htm`, the file the site's Terms of Use tab loads, read from its GitHub repository). So the weights' licence is **not verified as
permissive**: S8 offers "from scratch" and "pre-trained on the allowed sources" only (Lead call D62).

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
| D32 | 2026-11-25 | **Lead call** (planning lead, D4): job control is the package `app/jobctl/` with one queue for the whole app and one job at a time; job records persist in `<data_dir>/jobs/jobs.sqlite`, so a schedule and an unfinished job survive a restart. Extraction, SAM preview / estimate / run and playback copies all go through it; `app/jobctl` imports nothing labeling-specific (a test checks); `app/jobs.py` is removed in bite 2. | Spec §Job control: a separate worker process, survives the tab closing, reusable for training jobs; one queue keeps two GPU jobs apart (S1). Data folder, not the project, because the queue and the morning summary span projects. Nick can reverse it. |
| D33 | 2026-11-25 | **Lead call** (planning lead): a job is a worker process speaking `@@ {json}` lines (`loaded`, `item`, `done`, `paused`; S3's SAM worker emits `frame`, which bite 1b renames to `item`, and gains `paused`) and reading `pause` on stdin, or end of file when the app quits. Progress comes from the artifacts: a worker skips items whose output is on disk, so Resume runs the same params again. | S3's SAM worker already works this way and skips frames whose settings key is current; extraction already skips frames on disk (D19). One source of truth means no frame repeated or skipped after a crash. |
| D34 | 2026-11-25 | **Lead call** (planning lead): Pause ends the worker after the item in progress (the model is unloaded with the process); the app ends a worker that has not paused within 10 min. A job found running at app start becomes `interrupted` and the dashboard offers Resume or Cancel. Cancel keeps finished items. S3's Stop button becomes Pause. | Spec: pause frees GPU memory, a crash loses at most one frame, the app offers to resume. |
| D35 | 2026-11-25 | **Lead call** (planning lead): Estimate runs 1 warm-up + 5 frames taken spread over the frames Run would label and keeps their results; it replaces D26's preview-based estimate once it has run. Remaining time during a job = rolling average of the last 20 items × items left. | Spec numbers (1 + 5, rolling 20). Keeping the results means the estimate costs no extra SAM time. Nick can reverse it. |
| D36 | 2026-11-25 | **Lead call** (planning lead, SPEC Q10 memory guard): limits are one settings file for the laptop (`<data_dir>/jobs/settings.json`), copied into each job: device auto (NVIDIA, Apple GPU, CPU), threads half the cores, `nice` 10, pause between items 0 s, batch 1. The memory guard pauses a job when free memory is below a threshold the builder measures in bite 4 (during a tiled 4K run and a whole-frame 5.3K run) and proposes in the PR. The guard reads one named figure, which the builder picks in the bite 4 plan and measures and logs exactly as the guard reads it (on this laptop `vm_stat` "Pages free" showed 10.9 GB while `memory_pressure` showed 79 % free at the same time: QE PR 30); the lead logs the number here. A new library for reading free memory needs its licence read from PyPI metadata first; else `vm_stat`. | Measure, don't guess (D24): whole-frame SAM on 5.3K peaked at 26.7 GB Apple GPU memory on a 32 GB laptop, tiled 16.3 GB (S3 bench). |
| D36a | 2026-11-25 | **Lead call** (planning lead, D36): the memory guard reads psutil `virtual_memory().available` (free + inactive pages on macOS) and pauses below **2 GB** by default (adjustable on the Jobs page). | S4 bite 4 bench (PR 36, DESIGN §Bench results), real SAM 3 on the Apple GPU: lowest guard figure 3.5 GB on a tiled 4K run and 2.8 GB on a whole-frame 5.3K run; a 4 GB cut-off would pause an ordinary tiled run. `vm_stat` "Pages free" hit 0 GB in both, so it cannot be the figure. n = one 3-frame run per mode: re-check on a long run. Nick can reverse it. |
| D37 | 2026-11-25 | **Lead call** (planning lead): Start at / Stop by are clock times (next occurrence); the app must be running for them to fire, and the page says so. A start missed while the app was closed runs at the next app start if Stop by has not passed, else it shows "missed". Stop by pauses the job (Resume later), with reason "stop by". Offered on extraction and SAM Run, not on playback copies or Preview; SAM pre-screen (spec, deferred by SPEC Q6) gets Start at when it is built. | Spec §Overnight mode. A launchd service would fire without the app but is more machinery than the MVP needs. Nick can reverse it. |
| D38 | 2026-11-25 | **Lead call** (planning lead): keep awake = one macOS keep-awake process (`caffeinate -i -s -w <app pid>`) started by the app when a keep-awake job is scheduled or queued and ended (by its own PID only) when no such job is left. The reminder says plugged in and lid open; the page warns when `pmset -g batt` reports battery power. The app never changes power settings. | The man page on this laptop (read 2026-11-25): `-i` prevents idle sleep, `-s` system sleep, `-w` waits for a PID to exit, so a crashed app releases it. A scheduled job needs the laptop awake before it starts. Lid-closed sleep is not covered, hence the reminder; the builder checks the real behaviour through the app. |
| D34a | 2026-11-25 | **Lead call** (planning lead, Nick's S4 test: "not able to cancel the run and start a new one, I had to wait"): Cancel ends the worker at once (the frame in progress is dropped; finished frames stay); Pause still finishes the frame. Cancel also works while a job is pausing, and a pausing job says what it waits for. | Items are written under a temporary name and renamed, so killing a worker cannot leave a half-written item; only that item's time is lost. |
| D36b | 2026-11-25 | **Lead call** (planning lead, Nick's S4 test: "SHould it be on GPU instead?"): CPU fallbacks known to be harmless (first: `aten::_assert_async`) go to the job log only; any other fallback stays a warning on the job, worded "one step runs on the CPU (slower); SAM itself runs on the Apple GPU". | From the lead's note on Nick's screenshots (not re-measured by the builder): `aten::_assert_async` is a check with no maths; Nick's run took 67.7 s/frame against 63–66 s in the S3/S4 tiled 4K benches, and SAM used 14.46 GB of Apple GPU memory. |
| D36c | 2026-11-25 | **Lead call** (planning lead, Nick's S4 test: "it's pasuing itself"): a memory-guard pause is shown on the job everywhere with its figures and what to do; a job the memory guard paused resumes by itself once free memory has stayed above twice the limit for 60 s, so an overnight run does not stall until morning; the log and the summary record each pause and resume. A job paused by Nick, by Stop by or by Cancel never resumes by itself. | Twice the limit for a full minute keeps a job from pausing and resuming over and over while memory hovers near the limit. |
| D39 | 2026-11-25 | Third urchin type "black urchin" (SPEC Q18): a new class with the next free id, kind object, in `sam_group: urchin` (same prompt "urchin"), tagged `B` in label review. Its id is permanent; its name may change later. **Lead call** (planning lead): S5 lets a class be renamed with its id kept even after it has results, lifting S3's check that an id used in `sam_raw/` keeps its name (DESIGN §Planned for S3) for renames only; D6 keeps ids stable, and older exports' `classes.json` keep the old name. Nick can reverse it. Review is built to find outliers: `Tab` to the next untagged urchin with the viewer zoomed on it, a close-up panel, and "Tag all remaining as purple" per frame. | Nick, lead chat 2026-11-25: "we should add a third color for now / black, and later this name might change but it's certainly different from red and purple"; "most will be purple so I'd rather not need to click every single thing". Amends D18 (two types) and SPEC Q14 ("no third urchin class now"). The hourly packet's 2 urchin types are a later milestone's question. |
| D40 | 2026-11-25 | **Lead call** (planning lead): review state lives in `state.sqlite` (mask and frame review tables, action log for undo); `sam_raw/` gains only a stable `mask_id` per mask and hand-added masks (unit `added`). Export = `sam_raw` + review state. | Spec folder layout puts review state in `state.sqlite` and says re-export rebuilds from `sam_raw` plus the review state. Stable ids let a re-run keep reviewed masks (SPEC Q7). |
| D41 | 2026-11-25 | **Lead call** (planning lead, SPEC Q11): only accepted frames are exported; largest object % = largest object mask's pixel area / frame area; unknown = id 0; counts and cover from the final class map after review; classes not fully checked are listed per image (COCO `checked_classes`, a metrics column). | The packet's "largest single object as % of frame" is about how close an animal is: the mask measures that, a box overstates it. Id 0 was already set in D22. Nick can reverse it. |
| D42 | 2026-11-25 | **Lead call** (planning lead, SPEC Q9): the CVAT route is the COCO file with polygon outlines; nothing is imported back from CVAT in Build 1 (icebox). | Spec: no hand-editing of outlines in v1, export to CVAT for fine fixes; importing back is not asked for. The CVAT format claim is checked against CVAT's docs in bite 6. Nick can reverse it. |
| D43 | 2026-11-25 | **Lead call** (planning lead): splits group frames by site + dive date (from the file name, else the folder) and assign whole groups to come closest to 70 / 20 / 10 by frames, deterministically; with fewer than 3 groups the page warns and splits by video. | Spec: split by site and dive date so near-identical frames don't leak. A test set needs at least one group of its own. Nick can reverse it. |
| D44 | 2026-11-25 | **Lead call** (planning lead): export images are upright (copied as is when no EXIF rotation, else re-saved at quality 95); `seg_masks` are 8-bit PNGs of class ids. | D27 (COCO boxes and mask sizes must match the image files); JPEG quality 95 as for extraction (D17). |
| D45 | 2026-11-25 | **Lead call** (planning lead): area masks with straight edges at tile seams (Nick saw them on several S4 test frames) are fixed in bite 7 by comparing areas on the whole frame with tiles; the lead logs the pick. | `tiles.union_by_prompt` joins pieces but cannot add a piece the neighbouring tile missed, which leaves a straight edge. Areas are large, so the whole frame may be enough; measure first (time, memory: whole-frame 5.3K peaked at 26.7 GB, S3). |
| D46 | 2026-11-25 | **Lead call** (planning lead): "add a missed object" runs SAM 3's click / box prompt on the frame through the job module; if an add takes over 20 s the worker stays loaded while the Review page is open (10 min idle limit). | Spec §6. S1 showed SAM 3's click and box prompts work (3.1 lacks those weights, D10). One job at a time (D32). |
| D47 | 2026-11-25 | **Lead call** (planning lead, S5 bite 1, amends D31 and D40): a mask the reviewer deleted stays reviewed: it hides an unreviewed re-run mask of its own prompt / run unit at IoU > 0.5, and is itself never shown, counted or exported; masks of other classes at that spot are shown and follow D22 / D31 as usual (flagged on a clash). Among reviewed masks, kept beats deleted. Urchin type is stored as the mask's class id (purple, red or black urchin), not a separate column. | Spec §5: reviewed masks are never silently replaced, so a re-run must not bring back what the reviewer deleted. QE PR 44 (FAIL at `4e14d11`): first drafted for any class, which would hide a later real detection of another class; narrowed to the same prompt. Class ids already name the three types (D18, D39). Nick can reverse it. |
| D48 | 2026-11-26 | **Lead call** (planning lead, D45): area classes (rock, kelp) run on the whole frame scaled to 2016 px (`areas_whole_px: 2016` in `classes.yaml`; 0 = tiles as before); objects stay in tiles. A `classes.yaml` without the setting gets 2016 too, so existing projects switch as well and re-run their areas (only) on the next Run; setting 0 in the class form keeps tiles. | S5 bite 7 bench (PR 49), real SAM 3, 10 frames from 4 videos incl. 5.3K: seam pixels 1078 → 29 and 2110 → 91; 2–5 s per frame; peak Apple GPU 22.7 GB; whole-frame areas missed 2 of 10 frames (a rock-face close-up, a hazy algae patch) where tiles found only fragments. Nick can reverse it. |
| D49 | 2026-11-26 | **Lead call** (planning lead, S5 bite 7): the anemone prompt is "anemone" (was "sea anemone"); no confidence per class. | Same bench: urchins lost to an anemone mask under D22 rule 1: 27 of ~94 with "sea anemone", 6 with "anemone"; "sea anemone tentacles" took none but found almost no anemones. Anemone masks on urchins scored up to 0.93, so a per-class confidence would not separate them. The remaining 6 stay under the spec's "keep the higher score" and are fixed in review; whether urchin should beat anemone is a question for Nick. Nick can reverse it. |
| D50 | 2026-11-26 | **Lead call** (planning lead, S5 fix): a project made before D39 has no black urchin, so its Review page has no B key. The Review page names each starting urchin type the project lacks and offers one button, "Add black urchin", which appends that class to the end of the project's `classes.yaml` (id 10 if no class or SAM result ever used it, else the next new id) through the usual checks; nothing else in the file and no SAM result changes. A type counts as present by name, or by id in the same group (renamed). | Found while Nick tested S5 on an older project. Smaller than a class-form offer: one route and one note, on the page where the key is missing. Adding a class to the urchin group does not change the group's settings key, so no re-run is needed. Nick can reverse it. |
| D51 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S6 builder, bite 1): Build 2's tools live in their own Python environments under `tools_dir`, outside the app's environment and outside git. `tools_dir` is an optional key in `config.local.toml`, default `~/NereusData/urchin_detector_pipeline/tools`. `convert-venv` (Python 3.11) is built with `uv` from `app/fit/toolchain/requirements-convert.txt`, the exact pins of the earlier "gate" environment; Vela has its own small environment. A job worker runs in the app's environment and starts tool steps as child processes with the tool environment's Python. Model definitions live in `app/modeldefs/`: plain PyTorch that imports nothing else from `app`, so the conversion environment (torch 2.13.0, CPU) and the trainer (the app's torch 2.10.0, Apple GPU) build the same model from the same code; weights move between them as a `state_dict`. Third-party model code is a git checkout at a pinned commit under `tools_dir/third_party/`, put on `sys.path` by `app/modeldefs/`; it is not pip-installed, so its own requirements file is not applied. Setting the tools up is a job with a button (bite 3), not a terminal step. | The app pins torch 2.10.0 and numpy < 2 (D7, `pyproject.toml`); the known-good conversion route needs TensorFlow 2.19.0, onnx2tf 1.28.8 and numpy 2.1.3 (`pip freeze` of `~/nereus_ml/venvs/gate`), and Vela pins its own flatbuffers. Not on the SSD: it is exFAT (`diskutil info`), which cannot hold a Python environment's links. The earlier work imported YOLOX the same way (`yolox_urchin/model.py`). Measured: `convert-venv` 2.2 GB, each Vela environment 28 MB. SPEC constraint 7 (browser only). Nick can reverse it. |
| D52 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S6 builder): compilers. Every cell of the table states the compiler and its version. **N6:** ST Edge AI Core 4.0.0 (`v4.0.0-20500`), run as the earlier work did: OpenMV's `tools/modelc.py` inside the `firmware-builder` container with `~/openmv-sdk-1.6.0-linux-x86_64` and the OpenMV tree (`7d4dbf7`), now mounted read-only; Docker must be running. The paths and the image name are settings under `[fit]` in `config.local.toml`. **4.1.0 is not installed:** running ST's macOS installer was refused by the builder session's permission system (2026-11-27); its terms were read first (SLA0044 Rev 7; see `THIRD_PARTY_LICENSES`). Moving to 4.1.0 is one later change, compared against these numbers; it waits for Nick. Staying on 4.0.0 (SLA0104, already on the laptop from Nick's earlier work) is part of this Lead call: Nick's yes of 2026-11-27 names 4.1.0's terms. **AE3:** Arm Vela 5.0.0 from PyPI, run natively with OpenMV's `tools/vela.ini` and the arguments of `compile_model.sh` (`--system-config RTSS_HP_DTCM_MRAM --accelerator-config ethos-u55-256 --memory-mode Shared_Sram --optimise Performance`). Vela 5.2.0 is installed beside it for comparison only. **Pi:** the 8-bit TFLite file itself, run with LiteRT (`ai-edge-litert` 2.2.0, in `convert-venv`); ONNX Runtime and ncnn are tried in bite 3c; the pick is S7's, by measurement. | One variable at a time (CLAUDE.md value 5): the route that produced the measured boards' models is kept, and reproduces byte for byte (Bench results, S6 bite 1). OpenMV's current SDK (1.7.3: the `SDK_VERSION` file on `openmv/openmv` master) still bundles ST Edge AI Core 4.0 and Vela 5.0.0 (`openmv/openmv-sdk` master, `tools/downloads.cfg` and `tools/build.sh`; all read 2026-11-27), so 4.0.0 / 5.0.0 are what the cameras' firmware is built against. Vela 5.2.0 also places every operator on the NPU but estimates 12.2 ms where 5.0.0 says 28.1 ms and the board measured 24–26 ms (`STAGE1.md`). Nick can reverse it. |
| D53 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S6 builder): the `models/` layout in §Planned for S6. A trained model is `projects/<name>/models/<model id>/` (the spec's folder); an untrained candidate is `<data_dir>/fit/<arch key>/` with the same `recipe.json`, `int8/` and `targets/<camera>/` inside. `<model id>` is `m<number>-<arch key>`, numbered per project and never reused. `<arch key>` is readable and built from the recipe's architecture part: `<family>-<size>[-<variant>]-<input px>-c<classes>`, for example `yolox-nano-conv-256-c5`. | "What runs where" needs no project and no trained model (spec §First result), so untrained results cannot live in a project. The same inner layout lets one `convert` job serve both (spec stage 11). A trained model's placement is expected to equal its untrained sibling's but is compiled again, not assumed: the earlier trained export had 119 steps against 118 untrained. Nick can reverse it. |
| D54 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S6 builder): a recipe is one JSON file, `recipe.json`, in two parts. `arch` (all S6 needs, and what the arch key is built from): `kind` (detector / segmenter / instance), `family`, `size`, `variant` (for example `{"stem": "conv"}`), `input` [height, width], `classes` (the list of class names; untrained candidates use the five Build 2 classes, plus "other" for the segmenters), `outputs` (the raw output maps and how they decode), `code` {repo, commit}, `licence` and `licence_mark` ("Roboflow licence" or none). `train` (S8 fills it; absent for untrained candidates): dataset version, tiling, epochs, augmentation preset, starting weights, seed, library versions, device. Tiling sits in `train` / evaluation, not in `arch`: a tile is just an input of the `arch` size. | Spec: "model type × size × input size × tiling" is one recipe, and every run records what made it. Splitting it keeps the fit result tied to what the compilers see (architecture and input size) while training settings change freely. Nick can reverse it. |
| D55 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S6 builder): the untrained fit check uses seeded random weights and 20 seeded random-noise calibration images, not real frames. A trained model is calibrated with frames from its dataset version's training split (S11). | Spec stage 11: for the untrained check "calibration frames only set value ranges". The earlier gate calibrated with DUO frames; DUO is fenced (Nick, 2026-11-27), and S6 has no dataset version yet. Measured: with noise calibration the same architecture still places with 0 software steps (Bench results). Nick can reverse it. |
| D56 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S6 builder, bite 2a-ii): a recipe's `arch` part gains `export`: {`env`: the tool environment that builds and exports the model, `opset`: the ONNX opset, `simplify`: whether the ONNX file goes through onnxsim before the 8-bit step}. Default (YOLOX, the known-good route): `convert-venv`, 13, no. RF-DETR: `rfdetr-venv` (the conversion pins plus `rfdetr==1.9.4`; `app/fit/toolchain/requirements-rfdetr.txt`), 17, yes. `export` is not part of the arch key. For model code that is a pip package, `code.commit` holds the pin as `pip:<name>==<version>` instead of a git commit; the convert job checks the installed version against it. | RF-DETR is a pip package that needs `transformers`; adding it to `convert-venv` would change the known-good environment (D51, value 5). Its own exporter uses opset 17 (`rfdetr/export/_onnx/exporter.py`). Measured: onnx2tf 1.28.8 fails on RF-DETR's raw ONNX and converts it after onnxsim. Nick can reverse it. |
| D60 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S8 builder, bite 1; amends the spec's "frames by reference": frames are stored once per project by content hash, because the export is rebuilt each time): a dataset version is `datasets/v<N>/` with a `manifest.json`, its own copies of the export's label files, and frames named by content hash, stored once per project in `datasets/images/`. Versions are built under a temporary name, renamed, and never changed or removed by the app. The version's hash covers classes, label files, frames, groups and outside sources, not its number, name or date. | Spec stage 7: models train on a version, "never on whatever the export is today". The export is rebuilt from scratch each time (D41) and frames can be rejected later, so a path into `export/` or `frames/` is not a frozen reference; exFAT has no hard links. Labels are small, images are not: copy labels per version, images once. Nick can reverse it. |
| D61 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; proposed by the S8 builder, bite 1; a gap in the spec): splits across versions are carried by frame first (by file name, from the newest earlier version that holds the frame), then by group the same way; new groups are dealt toward 70 / 20 / 10 from the carried counts. When the export's grouping changes from video to site + dive date and a dive straddles splits, old frames stay where they were and the dive's new frames go to its most held-out split (test, then valid, then train); the Datasets page lists such groups. | Spec stage 7: one fixed test set; every group keeps its split. A frame that was ever a test frame must never become a training frame, or an older model's test score and a newer model's training set overlap silently; the other direction only shrinks training data. Nick can reverse it. |
| D62 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27): YOLOX's published COCO weights are not offered in S8. A run starts from scratch or from a model pre-trained in the app on allowed outside sources; `THIRD_PARTY_LICENSES` says why. | The repository states no licence for the weights apart from its Apache-2.0 `LICENSE`, and COCO's terms leave the images under Flickr's terms: the weights' licence is not verified as permissive (SPEC constraint 6; spec "Models that are safe to use": code, weights and data are three checks). Nick can reverse it. |
| D63 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27; stricter than the builder's proposal): RF100 underwater-objects is listed on the Datasets page but cannot be switched on; it shows "waits for Nick: same image lineage as DUO". Urchinbot is the default outside source. | RF100's own record says CC BY 4.0 and GO, but its images are URPC contest images, the lineage of the DUO set that Nick ruled out (2026-11-27). A licence question is Nick's; the lead batches it for him. Nick can reverse it. |
| D64 | 2026-11-27 | **Lead call** (spec lead, 2026-11-27): the "fully checked" rule stays as the spec has it. When a version would train on 0 objects of a chosen class because no frame is fully checked for it, the Datasets page says so in one line with what to tick on the Review page; Freeze still works. Nothing under `app/review/` changes. | Found on the real S5 project: no class was marked fully checked on any detector frame, so training would have seen nothing without a word of warning. Fewer clicks for marking is a Build 1 question the lead takes to Nick. Nick can reverse it. |

## Bench / test results

**S4 bite 4, 2026-11-25 — free memory while SAM runs (memory guard, D36).** M1 Max, 32 GB, SAM 3 on the Apple GPU,
default classes, limits 5 threads + low priority, guard off during the measurement. Sampled every 2 s by the
guard's own function (`limits.free_gb`) next to `vm_stat` and `memory_pressure`. CSVs:
`<data_dir>/mem/tiled_4k.csv`, `whole_53k.csv` (scratch data folder, not git).

| Run (3 frames) | s / frame | Peak Apple GPU | Guard figure (psutil available) before / typical / min | `vm_stat` free min | `memory_pressure` free min |
|---|---|---|---|---|---|
| 4K tiled (1008 px tiles) | 63–66 | 16.2 GB | 20.5 / 5.5–8 / **3.5 GB** (1 of 112 samples < 4 GB) | 0.0 GB | 23 % |
| 5.3K whole frame | 9–45 | 31.3 GB | 22.1 / 5–7 / **2.8 GB** (1 of 59 < 3 GB) | 0.0 GB | 16 % |

Proposed default threshold: **2 GB** (no pause in either run; a 4 GB guard would have paused the ordinary tiled
run). n = one run of 3 frames per mode: re-check on a long run.

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

### S5 bite 7 (D45): areas on the whole frame vs tiles; anemone prompts (2026-11-25)

Real SAM 3 on the Apple GPU, 10 frames from 4 videos (Caspar South 2025, Fullers 2022, Mendocino 2025 at
3840×2160; South Wreck 2024 at 5312×2988). Areas (kelp "kelp", rock/substrate "rock") three ways: tiles (as S3,
1008 px, 20 % overlap), the whole frame at full size, and the whole frame scaled to 2016 px wide (masks scaled back
up). *Seam px* = mask edge pixels lying exactly on an inner tile edge (a straight cut).

| Frame | Class | Cover % tiles / whole / whole 2016 | Seam px tiles / whole / 2016 | Seen on the overlays |
|---|---|---|---|---|
| Fullers 1 | rock | 4.6 / 19.1 / 19.3 | 59 / 29 / 16 | tiles: pieces; whole: the rocks |
| Fullers 2 | rock | 16.9 / 37.1 / 36.1 | **1078** / 51 / 29 | tiles cut the big rock into rectangles; whole covers it |
| Fullers 2 | kelp | 6.0 / 7.4 / 7.3 | 186 / 41 / 60 | same plants, whole is smoother |
| South Wreck 2 | rock | 19.5 / 68.7 / 63.3 | **2110** / 168 / 91 | tiles: square patches; whole: the boulder and rocks |
| Caspar 2 | kelp | 13.5 / 46.2 / 47.7 | 0 / 95 / 124 | whole also covers part of the reef top: mixed |
| Caspar 1 | kelp | 12.5 / 0 / 0 | 521 / 0 / 0 | tiles find a hazy algae patch; whole finds nothing |
| South Wreck 1 | rock | 6.1 / 0 / 0 | 580 / 0 / 0 | rock-face close-up: tiles find 2 square pieces, whole nothing |

The other 3 frames have no areas in any mode. Time for both area prompts: whole 2.0–4.8 s, whole 2016 2.1–3.1 s
(one image encoding); tiles share the 52–98 s per frame tile pass. Apple GPU memory after the whole-frame pass up
to 26.9 GB on 5.3K, after whole 2016 at most 22.7 GB (no higher than the tiles). Whole 2016 vs whole: IoU 0.88–0.97.
**Pick (Lead call D48):** areas on the whole frame at 2016 px (`areas_whole_px: 2016`; 0 = tiles); the two
misses are frames where tiles found only fragments. n = 10 frames.

*Anemone vs urchin* (tiles, after tile-edge duplicates; urchin masks at 0.4). Of the masks of each anemone prompt
at 0.4, how many sit on an urchin (IoU > 0.5) and how many score higher than that urchin (rule 1 then keeps the
anemone and the urchin is lost), on the 4 frames with urchins (94 urchin masks):

| Prompt | On an urchin | Beats the urchin | Elsewhere (all scores ≥ 0.3) |
|---|---|---|---|
| "sea anemone" (S3 default) | 85 | **27** | 187, scores 0.30–0.77 |
| "anemone" | 56 | 6 | 17, scores 0.30–0.52 |
| "sea anemone tentacles" | 30 | 0 | 1 |

Anemone masks on urchins score up to 0.93 ("sea anemone") and 0.82 ("anemone"), so a per-class confidence
cannot separate them. **Pick (Lead call D49):** prompt "anemone". A rule that urchin beats anemone whatever
the scores would contradict the spec's "keep the higher score" (D22 rule 1): not done; a question for Nick if
review shows urchins still lost.

### S6 bite 1: known-good reproduction of the earlier YOLOX-Nano placement (2026-11-27)

All on the laptop, no hardware. Outputs, logs and the three scripts used: `~/NereusData/urchin_detector_pipeline/s6/known_good/`.
Earlier numbers: `ADIN_SPI_OpenMV/ml/compile_gate_report.md`, `ml/yolox_urchin/STAGE1.md`,
`~/nereus_ml/runs/compile_gate_2026-08-22.json`.

| Step (one change each) | N6: ST Edge AI Core 4.0.0 | AE3: Vela | Same as earlier? |
|---|---|---|---|
| 1. The earlier untrained 8-bit file (`yolox_nano_256_int8.tflite`, sha256 `b7f4d37d…`), earlier container route | 118 steps: 116 hardware + 2 hybrid (DepthToSpace) + 0 software; 1,166,008 bytes | 5.0.0: one `ethos-u` operator; 998,608 bytes; 512 KiB memory; estimate 28.08 ms | byte-identical output files (sha256 `dd6e250a…`, `fea8049f…`) |
| 2. The earlier trained 8-bit file (`stage1_v1_256_int8.tflite`), same route | 119 steps: 117 + 2 + 0; 1,209,816 bytes | one `ethos-u` operator; 1,027,744 bytes; estimate 28.08 ms | byte-identical (sha256 `bd6bb274…`, `144d8bad…`) |
| 3. Untrained model exported again in the earlier environment (`venvs/gate`), noise calibration instead of DUO frames (D55) | 122 steps: 120 + 2 + 0; 1,190,584 bytes | one `ethos-u` operator; 1,006,368 bytes; estimate 28.08 ms | same placement class (0 software, 2 hybrid); the step count differs by 4 |
| 4. Step 3 in the new `convert-venv` (same pins) | — | — | ONNX and 8-bit TFLite byte-identical to step 3 |
| 5. Step 1's file, Vela 5.0.0 native on macOS (no container) | — | output byte-identical to the container's (`fea8049f…`); CPU operators 0, NPU operators 378 | yes |
| 6. Step 1's file, Vela 5.2.0 native | — | CPU operators 0, NPU operators 352; 995,184 bytes; 512 KiB; estimate 12.21 ms | placement yes; file and estimate differ (D52) |

The step count depends on the weights and the calibration, not only on the architecture (118, 119, 122 for the
same network), so the page's headline uses the software and hybrid counts, not the total. Also run: step 1's file
under LiteRT 2.2.0 on the laptop (int8 input 1×256×256×3, three int8 output maps; 2.0 ms per inference on one
thread, a laptop figure that says nothing about the Pi; `scripts/litert_run.py`, `litert/run.log`). Not done: ST Edge AI Core 4.1.0 (D52).

### S6 bite 2a-ii: RF-DETR Nano, first look (2026-11-27)

A probe by hand, not the `convert` job; the table's own numbers come from bite 3. RF-DETR Nano, untrained, 384 px,
5 classes, `rfdetr` 1.9.4; files and logs in `~/NereusData/urchin_detector_pipeline/s6/probe_rfdetr_nano/`.

| Step | Result |
|---|---|
| Build and run with every network connection refused (`socket.connect` patched to fail) | builds and runs; 0 connection attempts; 30,161,468 parameters; outputs (1, 300, 4) boxes and (1, 300, 6) logits |
| ONNX (opset 17, classic exporter) | 106.9 MB |
| onnx2tf 1.28.8 on the raw ONNX | fails in its `Expand` step |
| onnxsim 0.7.3, then onnx2tf | 8-bit TFLite, 31.3 MB (onnxsim's command-line summary crashes after it has saved the file; the job must call its Python function) |
| Vela 5.0.0 (AE3) | compiles; 3,789 operators on the NPU (92.8%), 293 on the CPU; needs 16,992 KiB of memory; output 35.1 MB (the model area is 24 MiB) |
| ST Edge AI Core 4.0.0 (N6) | does not convert: `INTERNAL ERROR: 'NoneType' object is not subscriptable` |
