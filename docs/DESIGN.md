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
- Tests: synthetic frames and a fake predictor (no torch, weights or real frames); the builder runs real SAM in
  the browser for bites 5 and 7.

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
