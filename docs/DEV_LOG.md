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

## 2026-11-28 — Sprint S7 — "CSI first" on an N6 card (D83c); a board_reset's text follows what was written (S7 builder)
**Branch:** sprint/7-n6-csi
**Done:** the card option "CSI first" (N6, a card that times something): a stage between the imports and `ml.Model`
with `ml/deploy_probe.py`'s camera set-up ("csi start") and one snapshot ("csi snapshot"), two stages; the input stays
the seeded image. Stage markers are now
numbered per window, and the fake N6 tells stages apart by what they do. A no-write card's `board_reset` no longer says
"the job put ROMFS0 back" (lead 2, card 7).
**Self-test:** fake N6: with CSI first a window runs import, csi snapshot, ml.Model, … in that order, the input's sha256
still the laptop's; a reset in the CSI stage names it ("stage 2 (csi start) … last stage that answered: stage 1
(import)"); the six-plus-one stages in CPython (a stand-in `csi`) give the laptop's sha256. Card page: see the PR.
**Broke/surprised us:** nothing.
**Next:** QE; the next card (the lead's): `fomo_face_detection.tflite` from /rom with CSI first, no write; then the ORT PR.

---

## 2026-11-27 — Sprint S7 — QE PR 126's nits: no board serial in agent.log; no reset for a board that never answered (S7 builder)
**Branch:** sprint/7-n6-nits
**Done:** the agent writes the N6's serial as `<n6_dfu_serial>` (another board's: its last 4 digits) in `agent.log`
and in every error it returns; `info` from the bootloader no longer returns it. `N6Board.reached` (and the fake
board's): a card whose board never answered ends without the hard reset and without a second error.
**Self-test:** tests only (no page changed): the log after a load and a wrong-serial refusal holds neither serial;
a card failing at `info` has no `finish` in `run.json` and no "hard reset" in its message.
**Broke/surprised us:** nothing.
**Next:** bite 3's ONNX Runtime on the Pi (TRACKER S7's next open builder item).

---

## 2026-11-27 — Sprint S7 — The N6 card ends with a hard reset (D82d); QE PR 123's nit (S7 builder, bite 4b)
**Branch:** sprint/7-n6-reset
**Done:** `N6Board.finish()` and the agent's `hard_reset`: `mpremote reset` on the N6's own port after the end-of-card
restore, on every exit the restore covers (done, Pause, a stop rule, another error) and after a restore-only card; not in
its bootloader; a failed reset is `no_answer`. A refused bootloader check says our N6 stays in its bootloader and how it
gets out. The end-of-card messages name the step that failed.
**Self-test:** no page changed beyond two hints. Tests on the fake N6 and the fake board: the last call is `mpremote
connect <Pyboard port> reset`, the AE3 untouched; a reset in DFU is refused; a failed reset ends the job after the
restore; a restore-only card ends with the reset. QE 10 (after PR 123): an scp to the Pi may take 60 s plus the file
at 100 kB/s (the 24 MiB backup: about 5 min) and its timeout is `no_answer`, not a crash; the agent sends alive lines
while it waits up to 60 s for the bootloader or the boot (a test with a 2 s laptop limit; it fails with them removed). QE 10's card 5 check: every mpremote and dfu-util call
is in `agent.log` (it was 0 bytes); Lead call D82e: the N6 images are stored once by sha256 in `bench/n6_images/`,
not in each run folder, and the backup is copied to the Pi only when it is not there yet.
**Broke/surprised us:** nothing.
**Next:** the next open S7 item.

---

## 2026-11-27 — Sprint S9 — D92a: a collapsed bootstrap shows Wilson on the objects (S7 builder 1, on S9)
**Branch:** sprint/9-d92a
**Done:** the lead's call D92a in `app/evaluate/metrics.py` (`_collapsed`): a zero-width bootstrap interval on
recall, small-object recall or precision becomes the Wilson interval on its objects (or predictions), labeled in
`interval_note`; AP50 and the means keep theirs with "bootstrap gave no spread". TRACKER boxes for QE 10's four
nits on PR 120 (the cache) and its note for bite 3 (an eval goes stale when best.pt changes).
**Self-test:** tests only (no page): 10 of 10 found gives recall 1.0 with Wilson [0.722, 1.0] (scipy 0.72247).
**Broke/surprised us:** nothing.
**Next:** S9 bite 3 (per-frame view), as the lead assigns.

---

## 2026-11-27 — Sprint S7 — The N6's DFU serial out of git (S7 builder, lead 2's ask)
**Branch:** sprint/7-n6-serial-config
**Done:** PR 123 had put the bench N6's real DFU serial in `app/bench/n6_board.py`, against SPEC's hard constraint (no
unit serials in git). It is now `[bench] n6_dfu_serial` in `config.local.toml` (placeholder in `config.example.toml`);
without it `open_board` refuses the N6, and `N6Board` refuses any load or restore before anything is sent. A test fails on
any run of exactly 24 hex digits in app/, tests/, docs/ or the example config. The history keeps the old line (rewriting
it is Nick's call).
**Self-test:** tests only (no page changed); the guard finds the old line's pattern and ignores sha256 values.
**Broke/surprised us:** my own slip in PR 123: I read the serial from the backup and wrote it as a constant.
**Next:** the N6 hard-reset PR (126) on top of this.

---

## 2026-11-27 — Sprint S9 — Bite 2b: how far a figure can be trusted (S7 builder 1, on S9)
**Branch:** sprint/9-intervals (PR 124)
**Done:** D92 in `app/evaluate/metrics.py`: a 95 % interval on every figure (Wilson for frame shares; a seeded
bootstrap, 1,000 draws, for the rest); whole site-and-date groups resampled from 5 groups up, frames below that with
a note; the rank floor per class (10 frames, 30 objects). `run.test_groups` reads the groups from the manifest (none
for a version split by video). QE 10's nit on PR 118: no class ticked on the Train page is red now.
**Self-test:** builder 2's m4 copy, metrics recomputed from its predictions: class 1 recall 0.941 [0.842, 1.0], AP50
0.941 [0.842, 1.0]; class 3 recall 1.0 [1.0, 1.0]; within ±20 % 1.0 [0.610, 1.0] (Wilson, 6 frames); 1 group, so
frames are resampled and the note says so; both classes under the rank floor (6 frames). Train page, every class
unticked: the span has class "problem", border rgb(198, 40, 40).
**Broke/surprised us:** nothing.
**Next:** S9 bite 3 (per-frame view), as the lead assigns.

---

## 2026-11-27 — Sprint S7 — Bite 4b: the N6 write route with a built-in restore of ROMFS0 (S7 builder, bite 4b)
**Branch:** sprint/7-n6-write (PR 123)
**Done:** `N6Board` (`app/bench/n6_board.py`) and the agent's `N6`: our ROMFS0 image built from the checked backup plus
ours (D82a), written with DFU alt 3 only, read back, booted, `/rom` checked; the timing script over `mpremote exec`; any
stop rule after a write may have started runs the restore from the backup itself (read back whole, `/rom` equal to the
backup's), or ends as the new `restore_failed`. QE PR 119 nits 1-3. Lead calls D82b (a restore-only card), D82c (every N6 card ends restored), D83a (the board's
input sha256 checked against the laptop's). A fake N6 (`tests/fake_n6.py`: fake mpremote and
dfu-util over a 64 KiB partition) for every test; nothing touched the rig.
**Self-test:** `tests/test_bench_n6.py` (14): load and time, three failed loads restored, a failed window restored, a
board that never comes back is `restore_failed`, a wrong backup sends nothing, alt 0-2 refused, `input_differs`, the
board script's sha256 equal to the laptop's. With the restore switched off 5 of them fail. Browser (scratch data
folder): a restore-only card on fake / n6 runs to "ROMFS0 restored 1×"; a 3 × 20 fake N6 card ends with the restore in
the rig log. On the laptop: the real backup passes the size, sha256 and re-encode checks, and our
image with `yolox-nano-conv-256-c5` is 11,195,160 bytes with the 10 vendor entries unchanged.
**Broke/surprised us:** an older test (an N6 card on a made-up real unit, ssh host `h`) reached `ssh h` once the N6 had
a board; `h` resolves to nothing here, and the test now uses the AE3. Open for card 4: whether an N6 that cannot boot our image comes back in DFU by itself.
**Changed before merge (QE 10 FAIL on 9467ea0):** the restore copies the backup itself; an N6 in its bootloader is a
normal start; only Pyboard ports are asked; dfu-util pinned to the backup's serial (`-S`, one device only); any other
error after a load ends restored too. QE 10's two probes pass on the fix.
**Next:** the restore-only card, then card 4 (the lead writes both; the rig session runs each after QE's review).

---

## 2026-11-27 — Sprint S9 — Bite 2a: the figures of an eval (S7 builder 1, on S9)
**Branch:** sprint/9-figures (stacked on sprint/9-evaluate, PR 121)
**Done:** `app/evaluate/metrics.py` (D91): count error (mean, mean absolute), within ±20 % and the strict share with
the frames passed by the one-object rule, empty frames with a prediction, precision, recall, AP50, small-object recall,
largest-box % error; each with its frames and objects; written as `metrics.json` by the evaluate worker. Bite 2
split: 2b is the intervals and the rank floor (D92).
**Self-test:** builder 2's m4 copy re-evaluated: class 1 recall 16 / 17, AP50 0.941, within ±20 % 1.0 (strict 0.833,
1 frame by the one-object rule); class 3 recall 10 / 10, AP50 1.0; together the 26 / 27 of bite 1's probe.
**Broke/surprised us:** the pycocotools check first disagreed (0.33 against 0.14): my test gave the boxes of one frame
the same annotation id, so COCO read one box several times; with one id per box they agree to 1e-6.
**Next:** S9 bite 2b (intervals, rank floor).

---

## 2026-11-27 — Sprint S9 — Bite 1: the evaluate job (S7 builder 1, on S9)
**Branch:** sprint/9-evaluate
**Done:** `app/evaluate/` (D90): job kind `evaluate`, one model × one test set → `eval/v<version>-<test hash>/`
(`predictions.jsonl` frame by frame, `eval.json` last); refusals (no best.pt with the Train page's reason, a
pre-training model, no predictor, no fully checked test frame, already evaluated); the eval scale from the recipe or
worked out for older models; tiles from `app/train/tiles.py` with a numpy NMS across seams; scores kept down to 0.01;
the YOLOX predictor on best.pt's EMA weights. Tests with a fake predictor (job, Pause / Resume, refusals, the
recipe's scale and cap carried) and a stand-in detector on painted squares (native boxes back in place).
**Self-test:** builder 2's m4 copy through `JobControl` on the Apple GPU: 6 test frames, factor 0.7586 worked out,
recall 26 / 27 and precision 26 / 26 at 0.3 (DESIGN §Bench results). No page: bite 3 adds the model list.
**Broke/surprised us:** QE 10 found a crash's cut last line in `predictions.jsonl` swallowed the next frame's line,
so a "done" eval could miss a frame silently. Fixed: whole lines on every start, every frame once before
`eval.json`; also one set of weights per eval and boxes clipped to the frame (QE nits 1, 2).
**Next:** S9 bite 2 (the figures, D91, D92).

---

## 2026-11-27 — Sprint S8 — Shrunk copies of outside images (S7 builder 1, on S8)
**Branch:** sprint/8-outside-cache
**Done:** `samples._load` with a cache folder: the smallest stored copy at least the size asked for, else decode
the original and store a copy at twice that size (not when that is the original's size); keyed by path, bytes and
mtime; whole files only. Pre-training samples name `<data_dir>/cache/outside_images/` (`yolox.outside_cache`).
**Self-test:** 64 Urchinbot photos (read only), probe scripts in my scratchpad: 0.093 s per image before; first pass
0.131 (writes 63 copies, 61 MB against 372 MB); then 0.023; epochs 1-3 at new scales 0.024-0.030 with 3, 2, 0 new
copies. Table in DESIGN §Bench results.
**Broke/surprised us:** the first version made no copy: JPEG's draft decode had already shrunk the image, and the
"as large as the original" check looked at that; it now uses the original's size. The trainer's outside-data test
wrote its copies one folder above its own: it now points the cache at its own folder.
**Next:** S9 bite 1 (the evaluate job, D90), on the tiling code.

---

## 2026-11-27 — Sprint S8 — The Train page's frame counts follow the class ticks (S7 builder 1, on S8)
**Branch:** sprint/8-live-counts
**Done:** one counter (`frame_counts` in `app/web/train_pages.py`) and one template (`_frame_counts.html`) for the
page and for `GET /p/<name>/train/counts?version=&classes=`; `train.js` fetches it on every class tick and swaps the
span (a late answer to an older tick is ignored); no class ticked says so; a bad class id or version says why.
**Self-test:** s8_squares copy, version 1: "With the classes ticked: 42 training frames in full, 12 validation
frames."; unticking sea star fetched `counts?version=1&classes=1` (200); unticking both: "Tick at least one class
…"; ticking sea star again: the counts back. (Every class is fully checked there, so the numbers do not move; the
test covers a version where they do.)
**Broke/surprised us:** nothing.
**Next:** shrunk copies of outside images (TRACKER S8).

---

## 2026-11-27 — Sprint S7 — Bites 3a, 3b, 4a and follow-ups (S6 builder 3)
**Branches:** sprint/7-bench-agent (PR 108), sprint/7-pi-runtime (112), sprint/7-pi-install (113), sprint/7-card-threads (115),
sprint/7-bench-nits (117), sprint/7-n6-romfs (119); also sprint/6-camera-order (109)
**Done:** the bench agent (JSON lines, LiteRT, alive lines), the ssh link only from a listed unit with the `ssh -G`
field-unit check, the Pi board for real units, `bad_answer` / `low_memory` / `install_failed` stop rules, `agent.log`;
the Pi's LiteRT environment from a hash-pinned file installed once as a logged rig change (D89); a Pi card's threads
(1-4); unique run folders and the other PR 105 / 113 follow-ups; the N6 model-area image read and written as OpenMV does
(D82a), the backup's image re-encoded byte for byte; Nick's camera order (Pi, N6, AE3) on every /fit table.
**Broke/surprised us:** backports.strenum 1.3.1 requires Python < 3.11, so the Pi resolves 1.2.8 (the rig session's dry run
gave the real versions; tqdm's piwheels file differs from PyPI's, hence PyPI only with hashes). The N6's vendor models are
now built with `atonn-v1.1.0` (firmware v5.0.1) while ours are `STAI-3.0.0` (SDK 1.6.0): D82a's gate did not pass; card 4
answers it on the board (Lead call (a)). The alive test was flaky because of a real fault, found by QE 9: the transport
waited on the pipe and then read a buffered stream, so an answer in the same chunk as an alive line was lost (`no_answer`);
now one reader thread hands over whole lines. I had first blamed Python's start time.
**Next:** S7 bite 4b, the N6 write route with a built-in restore from the backup (handoff: S6_builder3_handoff.md in the
lead's notes folder).

## 2026-11-27 — Sprint S8 — QE PR 111's nits: no best.pt yet, a capped overlap (S7 builder 1, on S8)
**Branch:** sprint/8-val-nits
**Done:** the Train page says when a model has no `best.pt` (taken from validated epochs only) and after which epoch
the first validation runs, and that a cancelled run never gets one; a run with `train.eval_overlap_capped` says
objects longer than the overlap − 5 px may be scored from a cut copy. S9's two halves of the nits are in TRACKER S9
bite 1.
**Self-test:** s8_squares copy: a 40-epoch run with "Validate every 20" cancelled after 2 epochs: the page shows the
note with "the first validation runs after epoch 20"; m5 (12 epochs, has best.pt, overlap 61 not capped) shows
neither note. First wording said "Resume it", which a cancelled run cannot: fixed.
**Next:** PR for the live frame counts (sprint/8-live-counts, done locally), then shrunk outside images.

---

## 2026-11-27 — Sprint S8 — Fixes 2: validation at D90's eval scale, in tiles (S7 builder 1, on S8)
**Branch:** sprint/8-val-tiles
**Done:** `app/train/tiles.py`: the eval factor (44 px over the training split's median shorter box side, clamped to
0.03-1.5: Lead call added to D90; raw and clamped recorded), the tile plan (overlap = largest training box at the
factor + 5 px, at most half a canvas), `cut_by_inner_edge`; `samples.frame_at`; the trainer validates whole frames at
that factor in tiles of the canvas, drops copies cut by an inner tile edge, NMS again across seams; the factor and
overlap are written to `recipe.json` at the first start and kept on resume (tested through the job); drawn validation
frames are whole frames up to 1024 px; the Train page's note says so.
**Self-test:** on a copy of S8 builder 2's synthetic `s8_squares` (scratch data folder): a 15-epoch run from the Train
page (Apple GPU) recorded factor 0.7586 (median side 58 px over 183 boxes), overlap 56 (run before the +5 px margin; 61 now), 72 tiles over 12 validation
frames, 56 objects (the old validation counted 34); the page shows four whole 486 × 364 frames. Builder 2's m4 (400
epochs, a copy) scored by the new validation: recall 0.98, precision 0.83 before the inner-edge drop, 0.95 after (old:
0.94 / 0.97 on 34 objects). Probe scripts in the session scratchpad.
**Broke/surprised us:** the first edge-drop version lost boxes lying whole but within 2 px of an inner edge (the
stand-in detector test caught it: recall 0.67): the overlap now adds that margin.
**Changed before merge (QE FAIL on 1b20435):** with the overlap capped at half a canvas, the inner-edge drop lost large objects in every tile: an edge-touching box is now dropped only up to overlap − 5 px, and the cap is recorded. Lead calls: the drop confirmed in D90; validation every `train.val_every` epochs (default 5) and the last, best.pt from validated epochs only.
**Next:** the other S8 open items in TRACKER, as the lead assigns.

---

## 2026-11-27 — Sprint S7 — Bite 2b: the `bench` job on the fake board (S7 builder 1)
**Branch:** sprint/7-bench-run (on top of 2a, PR 101)
**Done:** job kind `bench` (`app/bench/run.py`, `worker.py`): a reviewed card → `bench/<run id>/` with `card.json`,
`run.json`, `timing.jsonl`; Pause between windows, Resume skips what is on disk, Cancel keeps it; stop rules as job
errors with a hint; model files and the card re-checked by sha256 in the worker; "Run this card" and the runs table
on the card page. `FakeBoard` gained `fail_at` and `delay_s` for the Pause, Cancel and stop-rule tests. QE PR 101 nits: (1) a card for a
real unit must keep D87's "RESTORE.md matches the rig log", at save and at Run; (2) Run re-checks unit and model
files (the worker); (3) a refused card comes back with the form filled in; (4) an unreadable card file is a row,
not a 500; (5) DESIGN says the writer / reviewer check compares typed names only.
**Self-test:** scratch data folder, card 1 (fake / n6, 3 × 300): "Run this card" → "Run queued"; after a few seconds
the card page shows the run "done", 3 of 3 windows, median 10.00 ms, spread 9.97–10.01, 3,090,000 B free after load;
the folder holds card.json, run.json (outcome done, route fake, job 1) and 3 timing lines; Jobs shows the job done
"model_int8: median 10.00 ms over 3 windows"; the rig log has the fake load, before and after.
**Broke/surprised us:** nothing.
**Next:** S7 bite 3 (the bench agent and the Pi).

---

## 2026-11-27 — Sprint S8 — Bites 2a–5 as built; Estimate and validation frames; docs (S8 builders 1 and 2)
**Branch:** sprint/8-docs (docs), sprint/8-train-estimate (PR 87), sprint/8-view-rule (PR 86), sprint/8-pretrain (PR 90), sprint/8-boxes-only (PR 93)
**Done (first S8 builder, merged earlier today, logged here because its docs were held back):** bite 2a dataset
versions and the `freeze` job (PRs 63, 64); 2b the detector loader and "what training will see" (PR 65); 3 outside
data and starting weights as a recorded choice (PR 66); 4a the `train` job kind (PR 67); 4b the training samples
(PR 71) and the YOLOX trainer, on the CPU at that point (PR 72); 5 the Train page (PR 69).
**Done (S8 builder 2):** the Apple GPU fault found and fixed (PR 84, entry below). The Datasets view counts frames
by the trainer's rule through one shared function (PR 86). Bite 5 part 2 (PR 87): Estimate as a short job, the
validation frames with boxes under the curves, form fields that follow the model type. Bite 3's open half (PR 90):
pre-training on a version's outside data, and starting a run from such a model. S8's docs: TRACKER ticks
with what is open, DESIGN "As built" under §Planned for S8, D66 (which frames a run uses), the YOLOX import
dependencies in `THIRD_PARTY_LICENSES`.
**Broke/surprised us:** the OpenCV wheel that YOLOX's import needs carries other libraries' binaries, some under
the LGPL (its own `LICENSE-3RD-PARTY.txt`); recorded, laptop only. On the scratch project one epoch is one step
(42 frames, batch 32), so a 120-epoch run is 120 steps and proves little: the first run with recall well above 0
needed more epochs (numbers in PR 87).
A mistake of mine: PR 84 reached main with three conflict-marker lines in DESIGN (a second conflicted file missed
when merging main in); fixed by PR 89, which adds a test that fails on such markers.
The lead's call D66 (frames with only some classes checked train on their labeled boxes only) is built in PR 93,
with the lead's two further calls: a run with no fully checked training frame is allowed with a visible warning; a
run with no fully checked validation frame is refused.
PR 103 makes the job runner start after the job kinds are registered (TRACKER S8 "Fixes", item 1).
**Open in S8 (each has a box in TRACKER):** feeding outside images faster; mosaic; the haze preset and the warm-up
rule are not measured; a full pre-training run (hours); no run on Nick's real frames yet (no class is fully checked
on them, D64); the attribution text travelling with a model; three small carried QE nits.
**Next:** shrunk copies of outside images; then a full pre-training run on Urchinbot.

---

## 2026-11-27 — Sprint S8 — Fix: the job runner starts after the job kinds are registered (S8 builder 2)
**Branch:** sprint/8-runner-start (PR 103)
**Done:** `JobControl.start()`; the app creates the job module without starting it, registers every kind, then
starts it. Before, a job that was queued or due when the app started could be picked up in between and end "no job
kind … is registered" (QE 8). The tests that reopen a job folder use the same order.
**Broke/surprised us:** the fault showed as one schedule test failing now and then in other PRs' suites.
**Next:** TRACKER S8 "Fixes", item 2 (validation at the evaluation scale with tiles, D90).
## 2026-11-27 — Sprint S6 — Close: sidecar fix, PR 92's nits, the 4.1.0 line, as-built docs (S6 builder 3)
**Branches:** sprint/6-exfat-sidecars (PR 100), sprint/6-recipe-stale (PR 102), sprint/6-st41-page (PR 106),
sprint/6-close (this entry)
**Done:** Nick's data disk is exFAT, where macOS writes a `._<name>` file beside every file: the 8-bit step's
glob matched two files and every 8-bit step said "does not convert" (the Run all of 4:26 PM); `app/sidecars.py`
filters every wildcard glob on data_dir, with a guard test. PR 97 restacked twice onto main, with QE 8's two nits
(a changed `~/.stmai` is a job error). PR 92's nits: results record `made_with` and an older one says "Run
again". The N6's 4.1.0 result as its own line with a button. TRACKER: S6's generic boxes ticked, bite 1 reworded;
DESIGN §S6 as built; the Demo checklist under S6.
**Broke/surprised us:** the exFAT sidecars (any new glob in `app/` that starts with `*` must use
`sidecars.real`; the test fails otherwise). The 7 `tests/test_yolox_trainer.py` failures in this worktree's
venv are a missing `tabulate`, not code (QE ran them green).
**Not done:** 2d stays `[~]` (Nick's Roboflow plan check, PML section 4). QE PR 102 nit 2: at 5:24 PM PST Nick's
data folder held 21 results, all with an `int8.json` export record, so none is beyond the older-recipe check;
the lead's re-run was still writing results then.
**Next:** the lead hands Nick the S6 test build (Demo checklist above).

---

## 2026-11-27 — Sprint S7 — Bite 2a: test cards as data and a page (S7 builder 1)
**Branch:** sprint/7-bench-job
**Done:** bite 2 split in two by the builder: 2a the card (D84), 2b the `bench` job. `app/bench/card.py`: build with
every problem at once, save once under the next number, review in its own file against the card's sha256 (not by
its writer), list; `/bench/cards` (list + new card, D84/D87 stop rules prefilled) and `/bench/cards/<n>` (the card,
"Record the review"); links from the dashboard and the Rig log page. QE PR 98 nits: 1 (field units counted as
written), 2 (an "after" with no "before" is its own row), 3 (tests and self-test in their own data folders), 4 and 5
placed in TRACKER bites 3-5.
**Self-test:** scratch data folder, fake unit. The browser itself blocked windows = 2; card 1 saved with windows 3;
"S7 builder" as reviewer refused ("someone other than its writer"); "QE 8" recorded; the list shows the review.
**Broke/surprised us:** the page showed the model as "model_int8": every S6 candidate's file has that name, so a
model is now named by its candidate folder (tested with two candidates).
**Next:** S7 bite 2b (the `bench` job on the fake board).

---

## 2026-11-27 — Sprint S6 — Bite 2f: ST Edge AI Core 4.1.0 on the RF-DETR family (S6 builder 2)
**Branch:** sprint/6-st-4-1
**Done:** read-only check of Nick's install (v4.1.0-20732, Neural-ART module 12.1.0, macOS build only); the
known-good YOLOX Nano (117 + 2 + 0, as 4.0.0); the five RF-DETR full-integer files, then quantized ONNX for Nano
and Atto: none compiles (table in DESIGN §Bench results). `app/fit/n6_st41.py`: 4.1.0 as a second, guarded N6
compiler step with its tests; `[fit] stedgeai41`.
**Broke/surprised us:** ST's tool collects usage statistics. Its prompt showed after two runs of mine had already
gone with the network open (YOLOX Nano, untrained; no frames or weights are in what it collects). Reported at
once; Nick ruled the same hour (D71). Giving the tool a private HOME does not work (it ignores `$HOME`, and that
try queued one record, which the lead cleared); denying the write to `~/.stmai` in the sandbox does.
The install cannot go into the 4.0.0 route's container (no Linux `stedgeai`; `/Applications` is not shared with
Docker), and natively the relocatable step has no `arm-none-eabi-gcc`: placement only.
**Self-test:** the real step on YOLOX Nano (converts, 117 + 2 + 0) and Atto ("LESS … not supported"), with
`~/.stmai` unchanged by checksum both times (`s6/probe_st41/step_selftest/`).
**Not done:** the page part (a 4.1.0 line and button in the N6 rows): an open TRACKER box.
**Next:** S6's close: TRACKER ticks, DESIGN as-built, the Demo checklist, "Run all" on Nick's data folder.

---

## 2026-11-27 — Sprint S7 — Bite 1: bench units, the Board interface, FakeBoard, the rig log (S7 builder 1)
**Branch:** sprint/7-bench-core
**Done:** `app/bench/` (D81): `[bench]` units read from `config.local.toml` (field units refused by name or ssh host;
any unit not listed refused; no `[bench]` = the fake board only); `Board` with info / load / load-several / time /
reset and `FakeBoard` (seeded timings, memory, failures on request as the card's stop rules); the rig log
(`<data_dir>/bench/rig_log.jsonl`, "before" fsynced before the change, "after" with ok or the error); the "Rig log"
page at `/bench/rig-log`, linked from the dashboard; a commented `[bench]` example in `config.example.toml`.
**Self-test:** the app on a scratch data folder with no `[bench]`: the page says only the fake board can be used;
two fake loads (one done, one failed on request) are hidden until "Show them", then listed with their undo.
**Broke/surprised us:** QE failed it twice on the field-unit refusal (a mistyped or missing `field_units`, letter case, `user@host`, `:port`, an entry listed by its full host name): it now fails closed, with a test per case. No hardware touched; `[bench]` set only in tests' temporary settings with made-up names..
**Next:** S7 bite 2 (test card and the `bench` job).

---
## 2026-11-27 — Sprint S6 — Bite 2c-ii: RTMDet-Ins-tiny candidate (S6 builder 2)
**Branch:** sprint/6-rtmdet-ins
**Done:** licences read at the source first (mmdetection, mmcv, mmengine: Apache-2.0; `THIRD_PARTY_LICENSES`).
`app/modeldefs/rtmdetins.py`: RTMDet-Ins-tiny at 256 px, built untrained from `mmdet` 3.3.0 with the config's
pretrained backbone removed, raw output maps, its own tool environment `mmdet-venv` (in "Set up the tools").
Builder call D72: `mmcv-lite` plus a stand-in for `mmcv.ops` instead of compiling full mmcv.
**Self-test:** built through the app's code with every connection refused: 5,616,362 parameters, 0 network
attempts. Through the real convert job on the three cameras, 3 of 3 "done": N6 130 + 2 + 2 of 134, AE3 323 NPU +
1 CPU, Pi side runs; table in DESIGN §Bench results.
**Broke/surprised us:** `mmcv-lite` does ship `mmcv/ops/*.py`, only not what they load, so the first check for
"full mmcv is installed" was wrong; it now looks for `mmcv._ext`.
**Changed before merge (QE FAIL on 8481b10, two records):** D72 is a Lead call (made by the lead the same day).
The licence record was wrong to say nothing restricted is installed: `mmcv-lite` installs mmcv's operator folder as
source, with 13 NVIDIA-licensed files. Lead call: the set-up step removes `mmcv/ops` after the install and the
environment is "not ready" while it is there; done on the real tools folder, and the model builds the same
afterwards (5,616,362 parameters, 0 network attempts). Also: OpenCV's bundled LGPL libraries are named; the
"0 network attempts" is the by-hand build check's, not the job's; the build test is skipped by the suite and
`mmdet-venv` has no pytest, so that build is checked by hand (the PR says how).
**Not done:** SparseInst (time box). The page was not opened for this bite: it adds one candidate row and no page
code.
**Next:** 2f (the guarded ST Edge AI Core 4.1.0 step), S6's close.

## 2026-11-27 — Sprint S6 — Bite 2e: the RF-DETR family with RF-DETR's own export (S6 builder 2)
**Branch:** sprint/6-rfdetr-retry
**Done:** one retry with one thing changed (Lead call D70). By hand first (`s6/probe_rfdetr_tflite/`), then as the
family's 8-bit step in the convert job: `app/fit/to_int8_rfdetr.py` (RF-DETR's own export to TFLite, then
TensorFlow's full-integer converter on the SavedModel it leaves; network refused and counted), `arch.export.route`,
two Python 3.12 tool environments in "Set up the tools", the licence text written next to every Roboflow-licensed
result, and "how its 8-bit file is made" next to each RF-DETR candidate on the page.
**Self-test:** the six candidates × AE3, N6, Pi through the real job: 18 of 18 "done", 0 network attempts, numbers
equal to the by-hand probe; table in DESIGN §Bench results. Seen in the browser on those results: Detector / Pi
"runs, best: rfdetr-nano-384-c5"; Atto on the AE3 "does not run: compiles, but needs 4.2 MiB of memory against
3.9 MiB (free on the board in the earlier work, not re-measured; S7 measures it on the board); 355 of 2431
operators on the CPU"; Femto on the N6 "the compiler stopped with an internal error", error text folded.
**Broke/surprised us:** RF-DETR's exporter makes no full-integer file (it refuses), so the file the compilers get
is partly our making. On the Apple GPU torch 2.15 crashes inside RF-DETR's export; the step builds on the CPU.
The 256 px Nano's full-integer file makes LiteRT abort without a message; the Pi step now says so.
**Also:** the set-up job was run on the real tools folder: the two new environments, made by hand from the same
requirements files, were checked ("Checked 99 / 101 packages", nothing installed) and marked ready; done in 1 s
(`s6/selftest_2e/jobs_setup/1.log`). The two environments of the route before stay on the laptop, unused.
**Next:** 2c-ii (RTMDet-Ins-tiny), 2f (ST Edge AI Core 4.1.0), S6's close.

## 2026-11-27 — Sprint S6 — Fixes: a LiteRT fault on the Pi step is a job error; Run again (S6 builder 2)
**Branch:** sprint/6-pi-tool-fault
**Done:** QE's nit 1 on PR 79 (must-fix before the test build, Lead call): `pi_run` exits with its own code when
LiteRT cannot be imported or its operator-listing method is gone, and `pi.compile` raises the job error with a
fix ("set the tools up again") for that code. Before, any failure of the run wrote "does not run on the Pi".
A model LiteRT refuses is still a result. Nits 2 and 3: a comment at the private LiteRT call; the laptop note
says the peak memory is the whole Python process.
Also QE's nit 4 on PR 80 (filed to this builder by the lead): "Run again" cleared the candidate's shared 8-bit
file and only that camera's result, so the other cameras' cells described a file that was gone. Now it clears
every camera's result of that candidate, and the page queues the other cameras' cells that had a result again
behind it and says so.
**Self-test:** the real `pi_run` on `plainseg-s-256-c6` in `convert-venv`: still 30 operators, exit 0; with the
app's own Python (no LiteRT): exit 3, "tool fault: ModuleNotFoundError".
**Next:** PRs 80–83, bite 2e, 2c-ii.
## 2026-11-27 — Build 2 — S9 design (Build 2 lead 2)
**Branch:** sprint/9-design (docs only)
**Done:** DESIGN §Planned for S9 and Lead calls D90–D94 (evaluate job with the recipe's tiling, the figures and
the small-count rule, intervals and the paired rank rule, comparing on one test set with the leak flag, RF-DETR
Nano as a time-boxed laptop reference); TRACKER S9 split into 6 bites. Also today: PRs 92, 94, 96 merged on QE
PASS at their heads; Nick's answer 16 recorded (own instance segmenters for the N6).
**Broke/surprised us:** the lead's status log carried guessed clock times (QE caught it; corrected, `date` before
every time from now on).
**Next:** QE on this PR; S6's close and its Demo checklist; S7 bite 1 (S8 builder 3).
## 2026-11-27 — Build 2 — S7 design; lead hand-over (Build 2 lead 2)
**Branch:** sprint/7-design (docs only)
**Done:** took over as Build 2 lead at 3:46 PM (handoff `build2_lead_handoff_1.md` in the notes folder); merged PR 90
on QE PASS at `982e128`; Lead call D66 confirmed (partly checked frames train on their labeled boxes only). Wrote S7's
design: DESIGN §Planned for S7 and Lead calls D80–D86 (bench runner and agent on the Pi, load routes per board,
timing method, test card, Pi runtime order, power as an add-on), from the earlier work's load and timing methods
(read by a helper with file and line); TRACKER S7 split into 8 bites. SPEC "Waiting for Nick": three items added
from the handoff, the ST 4.1 installer item removed (Nick installed it).
**Broke/surprised us:** the two cameras hang on the Pi's USB, not the laptop, so the board side has to run on the
Pi; the earlier repo has no AE3 ROMFS build script and no Pi inference code.
**Next:** QE on this PR; then S7 bite 1 to a builder; S6's close and its Demo checklist.
## 2026-11-27 — Sprint S8 — Bite 4b (part 3): YOLOX trains on the Apple GPU (S8 builder 2)
**Branch:** sprint/8-apple-gpu
**Done:** found why YOLOX's gradients were wrong on the Apple GPU (PR 72's open fault): the trainer's batch was
channels-last in memory, and in YOLOX's training graph the Apple GPU then gives wrong weight gradients (torch
2.10.0 and 2.13.0; which operation is not isolated: a plain conv stack is right in both layouts, QE PR 84; DESIGN
D65 and §Bench results). `app/train/yolox.py` builds a contiguous batch, trains on the
GPU, and checks one small batch on both devices before every start (falls back to the CPU with a warning). A run
paused mid-epoch now keeps its loss sums in the checkpoint, so the epoch's "loss" is the whole epoch's mean (QE PR
72 nit 1). Self-test: 120 epochs on the scratch project through the Train page, 273 s, loss 17.65 → 4.42.
**Broke/surprised us:** the first probes on random images agreed between the devices, because `torch.rand` makes
a contiguous tensor; only the trainer's own batch showed the fault. The lead's fallback (a second environment
with another torch) would not have helped: 2.13.0 has the same fault.
**Next:** the Datasets view by the trainer's frame rule; bite 5 part 2 (Estimate, validation frames on the page);
S8's docs (TRACKER ticks, as-built DESIGN).
## 2026-11-27 — Sprint S6 — RF-DETR through the convert job (S6 builder)
**Branch:** sprint/6-rfdetr-results (docs only)
**Done:** the six RF-DETR candidates × N6 and AE3 through the real job (12 jobs, all "done"), and the Pi step by
hand on the three 8-bit files that were made. Table in DESIGN §Bench results.
**Result:** no RF-DETR size runs on any camera with today's route. Nano and Seg Nano: the N6's compiler fails; the
AE3's compiles them but the files are 33–37 MiB against a 24 MiB area with estimates of 1.5–13.5 s; plain LiteRT
cannot load them (`FlexErf`). Atto / Femto / Pico (Roboflow licence) do not convert with today's route: the 8-bit
step fails in onnx2tf 1.28.8, which is not a compiler verdict. So far the Roboflow licence gains nothing for the
cameras; the conv models are the ones with no software step on the N6 and no CPU operator on the AE3.
**Broke/surprised us:** Vela reports more memory for the 256 px Nano than for the 384 px one (not explained).
**Next:** one later try with RF-DETR's own exporter (onnx2tf 2.4 or newer), if the lead wants it; 2c-ii (RTMDet-Ins).
## 2026-11-27 — Build 2 — Lead's day: S6 and S8 through QE, planning locked (spec lead)
**Branch:** sprint/6-lead-lock (docs only)
**Done:** the spec merged (PR 60); builders for S6 and S8, a QE and a rig test session started; after the spec, 21 Build 2 PRs
merged on QE PASS at their exact heads (62, 67, 61, 63, 68, 64, 73, 65, 74, 71, 66, 75, 69, 76, 72, 77, 78, 79, 80,
81, 86). Lead calls confirmed in DESIGN: D51–D59 (S6), D60–D64 (S8) on main; D65 and D66 are in S8 PRs under review. First "what runs where" results (compilers on
the laptop, untrained models): YOLOX Nano / Tiny, the project's own plain conv segmenter and YOLOX with a mask
branch place with no software step on the N6 (2–3 hybrid steps) and no CPU operator on the AE3; torchvision's
segmenters do not; RF-DETR Nano fits neither OpenMV camera and no
PML size converts. Both first builders rotated with handoffs; successors run. Both cameras of the rig are backed up.
**Broke/surprised us:** the lead gave four sessions a session id it had made up from a folder name (QE caught it;
corrected within minutes). A blind "keep our side" merge would have dropped a licence entry (caught by a patch
compare before any push; builders do stack merges now). Stacked PRs that all edit `THIRD_PARTY_LICENSES`, DESIGN and
DEV_LOG conflict after every squash-merge: one PR per bite on main is cheaper than a stack. YOLOX on the Apple GPU
gave wrong gradients (a channels-last batch; fix in PR 84). A lead cannot start successor sessions while Nick is away.
**Next:** S6: PRs 82, 83, 85, bite 2e's results, RTMDet-Ins, the Demo checklist. S8: PR 84, a first run on real
frames, bite 5 part 2, docs. S7: the Pi image, QE on the backup evidence, then the bench runner's design.
What waits for Nick is listed in SPEC ("Waiting for Nick").

## 2026-11-27 — Sprint S6 — Bite 3d: "Set up the tools" (S6 builder)
**Branch:** sprint/6-tools-setup
**Done:** job kind `tools_setup` and a Tools section on `/fit`: each tool environment and the YOLOX checkout with
"ready" or the reason it is not, the N6 compiler's pieces (SDK, OpenMV tree, Docker image) as found / not found,
one button. An environment is ready only when it was finished by this job for the current requirements files (a
mark file with their hash), so a hand-made or outdated one is made again. A failed install stops the job with
`uv`'s last lines. 7 tests with stand-ins for `uv` and `git`.
**Self-test:** on Nick's real tools folder the page's status reads all four environments "not ready: not finished,
or made by hand" (true: the builder made them by hand) and the YOLOX checkout "ready"; SDK, OpenMV tree and
Docker image "found". Then the job itself on the real folder (after the RF-DETR run had finished): "done", 5 of
5, in 1 s; `uv` installed and removed nothing (the hand-made environments already matched the pins); all five
read "ready" afterwards and `convert-venv` still imports torch 2.13.0 and TensorFlow 2.19.0
(`s6/selftest_3d/jobs/1.log`).
**Next:** RF-DETR results into DESIGN; 2c-ii.
**Changed before merge (S6 builder 2, QE's nits 2–3):** the page says what "ready" means (at least the
requirements; hand-installed extras stay) and Docker is asked at most once a minute, not on every page load.

## 2026-11-27 — Sprint S6 — Bite 4b: several models on one camera, stored together (S6 builder)
**Branch:** sprint/6-stored-together
**Done:** `table.stored_together`: per camera, the best candidate of each kind that runs there (smallest file among
the best), each pair and all three summed against the room left: the 24 MiB model area minus the vendor's models
on the N6 and the AE3, the 512 MB of memory on the Pi. The vendor's share is derived from the earlier work's
images of 2026-08-25 (12.0 MiB on the N6, 8.3 MiB on the AE3; approximate, source in `table.py`). The AE3
time-box note is on the page.
**Self-test in the browser** (real results of bite 3b): N6: detector + segmenter + instance segmenter take 2.5 MiB
and fit with 9.6 MiB left; AE3: 2.1 MiB, 13.6 MiB left; Pi: "only one kind has a candidate that runs here so
far" (only one Pi cell had been run).
**Open, found here:** D51 says setting the tools up is a job with a button; it does not exist yet (TRACKER 3d).
**Next:** 3d, 2c-ii, RF-DETR's cells from the running job, and the stacked PRs through QE.
**Changed before merge (S6 builder 2, QE's nits 1–3):** the vendor's share is each earlier image minus the compiled
YOLOX files it holds (N6 12,577,188 bytes, AE3 9,269,500; read from `~/nereus_ml/romfs_*_stage1/`), not minus
the input files; only permissive candidates are picked; the Pi line says "files (in memory once loaded) against
512 MB of RAM".

## 2026-11-27 — Sprint S6 — Bite 4a: the "What runs where" page (S6 builder)
**Branch:** sprint/6-fit-page
**Done:** `/fit`: the table (model kind × N6 / Pi / AE3) from the `fit.json` files, a headline per cell and per
candidate (rules in `app/fit/table.py`: hybrid steps on the N6 count as "runs", as the measured YOLOX has two), a
detail table per cell, "Run all", Run / Run again, the page reloading while cells wait. Builder call in the
headline rules: a file larger than the model area is "does not run".
**Self-test in the browser** (the app on a copy of bite 3b's real results): the table shows "runs" for all three
kinds on the N6 and the AE3 with the best candidate named, the earlier-work lines with their date under Detector,
and "not tried" for the Pi. DeepLabV3 on the AE3 shows "does not run: its file (35.1 MiB) is larger than the model
area (24.0 MiB)"; LR-ASPP "runs, partly on the CPU" with its one bilinear resize listed. Pressing Run on the Pi
cell of `plainseg-s-256-c6` queued a job and the cell became "runs: 30 operators, all on the CPU, 91 KiB, memory
not measured (limit 512.0 MiB)". The three Roboflow-licensed sizes show "Roboflow licence: evaluation only".
**Broke/surprised us:** the first version made the 8-bit model again for every Run; now only Run again does.
**Changed before merge (S6 builder 2, on QE's nits 1–3 and two Lead calls):** the headline also compares the
memory a model needs with the camera's memory figure (AE3: the 4.09 MB free in the earlier work, a reference,
not re-measured; Pi: 512 MB; the N6 has no figure recorded yet and the page says "not checked"); a cell's best is
ranked among permissive candidates only, the best Roboflow-licensed one sits on its own line; a model that stops
at the 8-bit step reads "does not convert with today's route (8-bit step)", not "does not run"; a compiler that
stops reads "the compiler stopped with an internal error (tool, version)"; the error text is one click away;
"mJ gross". Seen in the browser on the real reports with RF-DETR's added: `rfdetr-nano-256-c5` on the AE3 reads
"compiles, too big for the model area: its file is 32.6 MiB of 24.0 MiB; needs 91.7 MiB of memory against
3.9 MiB …; 296 of 3757 operators on the CPU"; Atto / Femto / Pico read "does not convert with today's route".
Not done (QE nit 4): "Run again" on one camera still rebuilds the shared 8-bit model.
**Next:** bite 4b (several models on one camera: storage check; the AE3 time-box record); RF-DETR's cells from the
running job; 2c-ii.

## 2026-11-27 — Sprint S6 — Bite 3c: the Pi step (S6 builder)
**Branch:** sprint/6-pi-step
**Done:** `app/fit/pi.py` + `pi_run.py`: for the Pi Zero 2W the convert job runs the 8-bit TFLite once under LiteRT
(in `convert-venv`) and reports: runs yes / no, operator count (all on the CPU), any operator that is not built in,
file size, the 512 MB memory limit with its source. Time and peak memory measured there are stored under `laptop`
with a note; `estimate_ms` and `memory_bytes` stay empty, so the page cannot show a laptop number as a Pi number.
**Self-test (by hand on the 10 files of bite 3b's run):** all 10 load and run with only built-in operators; file
sizes 91 KiB (plain s) to 11.0 MiB (DeepLabV3).
**Not done:** ONNX Runtime and ncnn on the laptop. Each needs a different 8-bit file (ONNX QDQ; pnnx), so "the
same model on the three cameras" would no longer hold; they are S7's comparison on the bench Pi (D52).
**Next:** bite 4 (the "What runs where" page, storage check, AE3 time box); 2c-ii (RTMDet-Ins).

## 2026-11-27 — Sprint S6 — Bite 2d: the Roboflow-licensed sizes (S6 builder)
**Branch:** sprint/6-rfdetr-plus
**Done:** shown first, before any use: what `rfdetr[plus]` sends out (DESIGN §Bench results; the finding is also
in the sessions' shared notes). Then, on the spec lead's call (D59): `app/modeldefs/rfdetrplus.py` with Atto,
Femto and Pico at their stock sizes, licence "PML-1.0", mark "Roboflow licence" (a recipe without the mark is
refused), their own tool environment, the PML summary in `THIRD_PARTY_LICENSES` under "NOT PERMISSIVE".
**Broke/surprised us:** "untrained" is not offline by default for these sizes: with no RF-DETR weights named, the
package fetches the backbone's pretrained weights from the Hugging Face Hub. The block held (0 connections), and
the build now uses the package's own `force_no_pretrain`. The plan check TRACKER asks of the builder cannot be done
without Nick's account; it is with the spec lead for Nick.
**Next:** RF-DETR (all six) through the convert job; bite 3c (the Pi); bite 4 (the page).

## 2026-11-27 — Sprint S6 — Bite 3b: N6 and AE3 compilers in the convert job (S6 builder)
**Branch:** sprint/6-compilers
**Done:** `app/fit/n6.py` (ST Edge AI Core 4.0.0 through OpenMV's `modelc.py` in the container, everything but the
output mounted read-only), `app/fit/ae3.py` (Vela 5.0.0 native), `app/fit/reports.py` (Vela summary and log, ST's
"Epochs details", sizes, error lines), tested on excerpts of real reports in `tests/fixtures/fit/`. Limits with
their sources (24 MiB model areas from the boards' `board_config.h`). `[fit]` settings for the toolchain paths. A
missing tool (Docker not running, no SDK, no Vela) is a job error with a fix; a model that fails is a result.
**Self-test:** 10 candidates × 2 cameras through the real job, 20 of 20 "done", table in DESIGN §Bench results.
First findings: a detector, a plain conv segmenter and a conv instance segmenter (YOLOX + prototypes) all have
no software step on the N6 (2–3 hybrid steps each) and no CPU operator on the AE3; torchvision's MobileNetV3 segmenters do not (bilinear resize on the AE3's CPU; 20 software
steps with float convolutions on the N6; DeepLabV3 is too big for the AE3).
**Broke/surprised us:** nothing broke. Vela's estimate for DeepLabV3 on the AE3 is 1.4 s.
**Next:** bite 2d's definition (Atto / Femto / Pico) and RF-DETR through the job, 3c (the Pi), then bite 4's page.

## 2026-11-27 — Sprint S6 — Bite 3a: the convert job and the 8-bit step (S6 builder)
**Branch:** sprint/6-convert-job
**Done:** job kind `convert` on `app/jobctl/` (registered in the app; no page yet): two steps, "8-bit model" and
"compile", each skipped when its file is on disk. `app/fit/to_int8.py` runs in the model's tool environment: seeded
build, ONNX, optional onnxsim (Python call), noise calibration, `onnx2tf -oiqt`, `int8.json` with versions, the
checked code version and the file's sha256. `fit.json` per model × camera; a failed 8-bit step or compile ends
"done" with `converts: false`; a missing tool environment is a job error with a fix. `tools_dir` in
`config.local.toml` (optional). 9 tests with a fake 8-bit step and a fake compiler.
**Self-test (real 8-bit step through the job, fake compiler):** `yolox-nano-conv-256-c5` done in 49 s, 8-bit file
1,163,360 bytes; `plainseg-s-256-c6` in 15 s, 92,912 bytes; outputs in
`~/NereusData/urchin_detector_pipeline/s6/selftest_3a/`.
**Broke/surprised us:** onnx2tf leaves float and half-quantised copies several times the model's size (340 MB for
RF-DETR); the step removes them.
**Next:** bite 3b (N6 and AE3 compiler modules, parsers tested on the known-good run's real reports).

## 2026-11-27 — Sprint S6 — Bite 2c-i: instance-segmenter candidates (S6 builder)
**Branch:** sprint/6-instance-candidates
**Done:** RF-DETR-Seg Nano in `app/modeldefs/rfdetr.py` (same package and licence as the detector; input a multiple
of 12, stock 312 px; outputs 100 boxes, logits and 78 × 78 masks; 33,271,645 parameters) and
`app/modeldefs/yoloxproto.py` (YOLOX Nano / Tiny plus 16 prototype masks and 16 coefficients per box; 974,382 and
5,204,566 parameters), both built untrained in the tool environment (D58).
**Broke/surprised us:** YOLACT's own models are not small (ResNet / Darknet-53 backbones), so the "small conv
instance segmenter" is our own head on YOLOX for now; RTMDet-Ins-tiny is split off as 2c-ii because it needs the
mmdetection stack.
**Next:** bite 2d (what `rfdetr[plus]` sends out, before any use), then bite 3a (the convert job).

## 2026-11-27 — Sprint S6 — Bite 2b: segmenter candidates (S6 builder)
**Branch:** sprint/6-segmenter-candidates
**Done:** `app/modeldefs/tvseg.py` (torchvision LR-ASPP and DeepLabV3 on MobileNetV3-Large, BSD-3-Clause read at
the source, built with no weights) and `app/modeldefs/plainseg.py` (our own plain conv net, sizes `s` and `m`);
six classes (the five plus "other"), 256 px, one raw scores map (D57). These need only torch and torchvision, so
the app's own tests build all four and check the output shapes: LR-ASPP 3,218,988 parameters, DeepLabV3
11,021,622, plain `s` 53,030, plain `m` 126,966. A test proves the plain net holds only conv, batch-norm, ReLU6
and nearest upsampling.
**Broke/surprised us:** nothing.
**Next:** bite 2c (instance segmenters: RF-DETR-Seg Nano and one or two small conv ones).

## 2026-11-27 — Sprint S6 — Bite 2a-ii: RF-DETR Nano candidate (S6 builder)
**Branch:** sprint/6-rfdetr-nano
**Done:** `app/modeldefs/rfdetr.py`: RF-DETR Nano (Apache-2.0, read at the source) as a detector candidate at 384 px
(stock) and 256 px, built untrained from the `rfdetr` 1.9.4 package in its own tool environment (`rfdetr-venv`,
`requirements-rfdetr.txt`); per-candidate export settings (D56). Shown before anything else: with every network
connection refused, the model builds and runs with 0 connection attempts (weights are only fetched when a weights
file is named; Nano loads no DINOv2 weights). First look by hand (DESIGN §Bench results): converts to 8-bit only
after an onnxsim pass (31.3 MB); Vela puts 92.8% of operators on the NPU but it needs 17 MB of memory and a 35 MB
file; ST Edge AI Core 4.0.0 stops with an internal error.
**Broke/surprised us:** RF-DETR's input must be a multiple of 32 (its backbone refuses 336). onnxsim's command line
crashes while printing its summary, after saving a good file. `rfdetr` ships its own TFLite exporter, which asks
for onnx2tf 2.4.0 or newer on Python 3.12 (its package metadata); the known-good route here is 1.28.8, so that
exporter is a later one-variable try if the table needs it.
**Next:** bite 2b (segmenters: torchvision LR-ASPP and DeepLabV3 on MobileNetV3, and one plain conv net).

## 2026-11-27 — Sprint S6 — Bite 2a-i: model definitions, YOLOX candidates (S6 builder)
**Branch:** sprint/6-detector-candidates
**Done:** `app/modeldefs/` (plain data + PyTorch, imports nothing else from `app`; a test checks): candidates as
the `arch` part of a recipe (D54), `arch_key` folder names (D53), checks that say what is wrong, `build()` for the
exported module. YOLOX Nano / Tiny (plain first layer, three raw maps) at 256 and 320 px with the five Build 2
classes, adapted from the earlier `yolox_urchin/model.py`. YOLOX is an unmodified checkout at `6ddff48` in
`<tools_dir>/third_party/YOLOX` (LICENSE sha256 `0ec3668d…`, Apache-2.0). Self-test in the conversion environment:
all four candidates build; outputs (1, 10, px/8 … px/32); 896,238 parameters (Nano), 5,032,086 (Tiny).
**Broke/surprised us:** importing YOLOX needs `loguru`, `tabulate` and `cv2`; the app's environment has none of
them, so the two build tests skip there (S8's trainer will need them if it trains in the app's environment).
Cloning YOLOX on macOS warns about two demo files whose names differ only by case; the model code is not affected.
**Next:** bite 2a-ii (RF-DETR Nano), then 2b (segmenters).

## 2026-11-27 — Sprint S8 — Bite 1: prior art, licences, `datasets/` layout (S8 builder)
**Branch:** sprint/8-prior-art (docs only)
**Done:** re-read `ADIN_SPI_OpenMV/ml/yolox_urchin/` at `994b8c2` (trainer, model, data, tests, export, decode,
eval, model card) and wrote what S8 reuses (DESIGN §Planned for S8): the training loop and its numbers, the
sample / augmentation code, the blur tests; not its control page (the job module replaces it), not its corpus or
weights (DUO). YOLOX's licence read at the source and added to `THIRD_PARTY_LICENSES` with the starting-weights
record. `datasets/` layout and the split-carrying rule logged as D60 and D61.
**Broke/surprised us:** the export is rebuilt from scratch on every Export, so "frames by reference" cannot point
into `export/`: frames are stored once per project under their content hash instead (D60). YOLOX's published COCO
weights carry no licence statement of their own and COCO's images stay under Flickr's terms: not offered. The
earlier work left haze augmentation out on purpose (it hurt Nano); TRACKER 4b asks for it, so 4b measures it.
RF100 (verdict GO, CC BY 4.0) shares DUO's URPC image lineage: raised with the spec lead before bite 3.
**Later the same day (code in draft PRs, not merged; their DESIGN / TRACKER notes follow each merge):** bite 2a =
PR 63 (`app/datasets/versions.py`, `splits.py`) + PR 64 (Datasets page, `freeze` job); 2b = PR 65 (loader and "what
training will see"); 3 = PR 66 (outside data: Urchinbot and RF100 allowed, DUO and the rest refused with reasons;
starting weights as a recorded choice); 4a = PR 67 (`train` job kind with a fake trainer, no torch). Self-tests in
the browser on a scratch copy of the S5 project, data folder `~/NereusData/urchin_detector_pipeline/scratch_s8_builder/`.
**Also seen:** on that real project no class is marked "fully checked" on any detector frame, so by the spec's rule
training would see 0 objects: the demo needs those boxes ticked. The full suite must run at normal priority: under
`nice -n 10` the job tests time out (319 passed without it).
**Next:** bite 4b (YOLOX trainer) once S6 bite 2a's `app/modeldefs/` YOLOX definition exists; bite 5 (Train page).
## 2026-11-27 — Sprint S6 — Bite 1: prior art, toolchains, known-good reproduction (S6 builder)
**Branch:** sprint/6-toolchains (docs, licence records and one requirements file; no app code)
**Done:** read `ADIN_SPI_OpenMV/ml/` (export, compile scripts, gate report, STAGE1) and `~/nereus_ml/`; what is
reused is a table in DESIGN §Planned for S6. Known-good first: the earlier untrained and trained YOLOX-Nano 8-bit
files recompiled with the earlier container route are byte-identical to the earlier outputs on both compilers (N6
116 + 2 + 0 of 118 and 117 + 2 + 0 of 119; AE3 one `ethos-u` operator). Then one change at a time: export again with
noise calibration (0 software steps still); a new `convert-venv` with the same pins (byte-identical files); Vela
5.0.0 native on macOS (byte-identical to the container). Numbers: DESIGN §Bench results. Decisions D51–D55: tool
environments under `tools_dir`, compilers, `models/` and `fit/` layout, `recipe.json`, noise calibration for
untrained candidates. Licences of every tool in `THIRD_PARTY_LICENSES`; ST's terms read (4.0.0: SLA0104; 4.1.0:
SLA0044) and saved under `tools/licences/`. Three stale "Nick merges the spec" lines corrected (PR 60 is merged).
**Broke/surprised us:** ST Edge AI Core 4.1.0 is not installed: the builder session's permission system refused
running ST's installer. OpenMV's current SDK still bundles 4.0 and Vela 5.0.0, so the known-good versions are also
what the cameras' firmware is built against. The N6 step count changes with weights and calibration (118 / 119 /
122 for one network). Vela 5.2.0 halves the time estimate (12.2 ms against 28.1 ms; the board measured 24–26 ms).
The SSD is exFAT, so tool environments live on the laptop's disk.
**Next:** bite 2a (detector candidates: YOLOX Nano / Tiny, RF-DETR Nano as model definitions in `app/modeldefs/`).

## 2026-11-26 — Build 2 — Spec written with Nick (spec lead)
**Branch:** sprint/6-build2-spec (docs only)
**Done:** `docs/source/build2_spec.md` (train, evaluate, compare, fit to the cameras, bench test, in the same app);
SPEC goal, inventory, hard constraints 1, 6 and new 9, non-goals, Q19–Q24; TRACKER ladder S6–S11. Nick answered seven
questions in the spec lead's chat (instance segmentation laptop first; permissive only; test frames may go to his bench
units; two of his rigs as bench units, he runs bench tests; frames from file first; outside data from his earlier work).
Prior art read: `ADIN_SPI_OpenMV/ml/` (YOLOX on both NPUs, measured), its `pi/hil/` rig and dataset licence records.
Licences and toolchain facts carry "lead" (read at the source by the spec lead) or "helper" (to be re-read by the
builder) in the spec.
**Broke/surprised us:** hard constraint 1 ("images never leave the laptop") ruled out any bench test with real frames
until Nick allowed bench units. ST's own zoo models may run only on ST chips, so they cannot be the shared model.
Ultralytics is AGPL for trained models too. The Pi has no inference code in any of Nick's repos yet.
**2026-11-27, Nick's comments:** "what hardware can run each type of model" comes first, so the ladder is reordered
(S6–S7 the "what runs where" table for three kinds × three cameras, untrained models; training from S8); RF-DETR Nano
and the sizes below it become candidates, the PML sizes for evaluation only; the table also answers whether a camera can hold and run several models at once. Nick has a Roboflow
account and agreed to its terms, so the PML sizes can be tested (no email to Roboflow). Power monitoring on the Pi 5
is an optional bench metric he wires later; the software does not wait for it. Nick wants to be left out of the
build and test cycle: one team test session owns the Pi Zero 2W rig, backs it up fully first (Pi image over SSH, the
cameras' flash, model area and firmware), may then change anything, and restores it after. Training covers sea stars and urchins only for now. Read at the source: PML 1.0 needs an
accepted Roboflow platform plan and allows usage tracking; RF-DETR-Seg is Apache-2.0 at every size.
**Next:** S6 bite 1 (the spec PR, 60, was merged on 2026-11-27 as `c0e69f3` by the spec lead for Nick, answer 15).

## 2026-11-26 — Sprint S5 — Fix: add a missing urchin type (D50)
**Branch:** sprint/5-add-urchin-type
- Done: projects made before D39 have no black urchin, so the Review page had no B key (found in Nick's S5 test).
  The Review page now names each missing starting urchin type with one button, "Add black urchin": it appends the
  class to the end of `classes.yaml` (id 10 if unused, else the next new id) with the usual checks; the other classes
  and SAM results stay as they are. `classes.missing_types` / `add_type`, route `POST /p/<name>/labels/add-type`,
  tests `tests/test_missing_type.py`; full suite 296 passed.
- Self-test: a scratch copy of the S5 project with the black urchin line removed showed the note; the button
  appended id 10 at the end, the note went away, and the frame's panel showed "black urchin (B)".
- QE PR 58 nit fixed: a type removed after it had SAM results gets back the id its results hold (a new id would be
  refused by the D6 check); test added, full suite 297 passed.
- Broke: nothing seen. Next: Nick clicks the button on his older project.

## 2026-11-26 — Sprint S5 — Bites 4b, 5b, 6a, 6b (builder, overnight)
**Branch:** sprint/5-fast-tagging (PR 54), 5-add-on-page (PR 55), 5-export-page (PR 56), 5-licences-readme (PR 57)
**Done:** fast urchin tagging on the Review page (Tab zooms to the next untagged urchin, P / R / B tag and move on,
close-up panel, "Tag all remaining as purple"); add a missed object on the page (click or drag a box → a `sam_add`
job; real SAM added a missed urchin, score 0.94); the Export page (Export as a job, checks PASS / FAIL, splits, the
D43 warning, per-class counts); `THIRD_PARTY_LICENSES`, README, the CVAT check (docs only, not tested end to end),
9 of 10 acceptance boxes ticked with evidence. Lead calls D47–D49 logged by the lead (PR 53).
**Broke/surprised us:** the close-up `<img>` took the viewer's absolute-position rule and covered the panel; a
clicked button kept focus so Enter could press it twice; the Export page did not refresh after the job until
`data-reload-on` was added. Stacking PRs on merge commits made rebases slow: prefer one parent per PR.
**Next:** the lead hands Nick the S5 test build; the open box is his `frame_metrics.csv` spot check on 5 frames.

## 2026-11-26 — Sprint S5 — Bites 3a, 3b, 5a (builder, overnight)
**Branch:** sprint/5-review-page (PR 50), 5-review-checks (PR 51), 5-add-object (PR 52)
**Done:** the label Review page (`/p/<name>/labels`): frame list, S2's viewer reused with a click hook, picture
modes, click to select, class picker, X / Enter / Z, collapsible panel; split check, fully checked per class, running
totals over accepted frames; add-a-missed-object backend (SAM 3 click / box on a crop, job kind `sam_add`). Browser
self-tests on the scratch copy with real SAM results; a real-SAM click: 16.9 s per add (D46: no kept worker).
**Broke/surprised us:** the browser pane's viewport (1024x768) is not the screenshot frame (800x600): click by
element reference, not by screenshot coordinates. PRs 47 and 50 both add an as-built bullet at the same place in
DESIGN, so the second to merge needs a rebase. QE PR 44: a deleted mask hid another class (fixed: own run unit only).
**Next:** bite 4b (P / R / B, Tab zoom, close-up, tag all) on the Review page; 5b (click / box on the page); 6.

## 2026-11-25 — Sprint S5 — Bites 1, 2, 4a, 7 (builder, overnight)
**Branch:** sprint/5-review-state (PR 44), 5-export (PR 46), 5-urchin-tagging (PR 47), 5-sam-quality (PR 49)
**Done:** review state in `state.sqlite` with stable mask ids, final masks after review, accept rule, undo (bite 1);
export writers + splits + metrics + checks (bite 2; new dependency contourpy for COCO outlines); black urchin, renames
that keep the id, tagging helpers (bite 4a, backend of bite 4); areas on the whole frame at 2016 px and the anemone
prompt, from a 10-frame real-SAM bench (bite 7, Lead calls proposed). Self-tests on a scratch copy of an S4 project
(16 real 4K frames): export checks PASS, polygon IoU median 0.997; rename re-runs nothing; Preview re-ran areas only.
**Broke/surprised us:** a run unit's key is the class name, so a rename would have orphaned its results
(`follow_renames` re-keys on save). macOS `._*` files on the exFAT SSD tripped the split check. "sea anemone" takes
27 of 94 urchins under rule 1; whole-frame areas miss on 2 of 10 frames. QE: a deleted mask hid another class (fixed).
**Next:** bite 3 (Review page) now that S4's page PRs are merged; then 4b, 5, 6.

## 2026-11-25 — Sprint S4 — Closed (planning lead)
**Branch:** sprint/4-close (docs only)
**Done:** S4 marked `[x]`. Nick passed the checklist at 10:13 PM (lead chat). His feedback from the test went in as
bites 6 (PR 41: Run panel order, Cancel at once, harmless CPU fallback in the log), 6b (PR 42: Resume, progress,
pages in sync, memory-guard message and auto-resume, project page) and 6c (PR 45: "Labelled so far", project row
from disk); Nick re-checks them in the S5 test build. Lead calls D34a, D36a–D36c.
**Broke/surprised us:** the memory guard paused Nick's run while he used the laptop (1.8 GB free); the reason was in
the log only, now on every page, with auto-resume.
**Next:** S5 bites, overnight.

## 2026-11-25 — Sprint S4 — Bite 6c: Labeled so far, project row from disk (builder)
**Branch:** sprint/4-labeled (stacked on sprint/4-feedback-b)
**Done:** Nick's items 9, 9a, 8a: a "Labeled so far" grid on the Labeling page (newest first, live while Run goes
on, legend and large viewer shared with the preview), and the project page's SAM labeling row counted from disk.
Self-tested with real SAM: a new frame appeared first in the grid during Run with no page reload.
**Broke/surprised us:** the large viewer only existed with a preview; it is now always on the page. I first
numbered this "bite 8"; renamed to the lead's 6c.
**Next:** QE on PRs 42 and 45; the lead closes S4 after they merge.

## 2026-11-25 — Sprint S4 — Bite 6b: live job boxes, Resume, memory-guard auto-resume (builder)
**Branch:** sprint/4-feedback-b (stacked on sprint/4-feedback)
**Done:** Nick's items 4–8: one job panel and banner (`_job.html`) on every page, live fragments + `live.js`
instead of reloads, progress bar with a running sign and "done by", Resume next to the state and in a banner,
memory-guard reason on the job and auto-resume (D36c), job status on the project page. Self-tested with real SAM:
paused from the Jobs tab, the Labeling tab followed with no reload; resumed from the banner, it followed back;
a 40 GB limit gave "Paused by the memory guard: 22.6 GB free (limit 40 GB)" on the page.
**Broke/surprised us:** Preview / Run stayed greyed after a pause until reload: buttons now follow the job
(`data-job-lock`). A background tab stops polling and catches up within 2 s when shown.
**Next:** QE on PR 42; then bite 6c (Labelled so far, project row).

## 2026-11-25 — Sprint S5 — Plan (planning lead)
**Branch:** sprint/5-plan (docs only)
**Done:** S4's test checklist passed (Nick, lead chat, 10:13 PM: "I've tested everything in the check list and it
passes"); his feedback items are S4's last bites, re-checked in the S5 test build. S5 planned under CLAUDE.md
rule 7: seven bites in TRACKER S5 (review state, export writers, review page, urchin types and fast tagging, add a
missed object, export page + licences + README, SAM quality), DESIGN §Planned for S5, Lead calls D40–D46 (SPEC Q9,
Q11). Nick saw straight edges on area masks at tile seams on several frames: bite 7 (D45).
**Broke/surprised us:** there is no page yet to see a finished run's frames; S4's "Labelled so far" item covers it.
**Next:** QE on the plan; the S5 builder starts bite 1 tonight.

## 2026-11-25 — Sprint S4 — Bite 6: Nick's test-build feedback (builder)
**Branch:** sprint/4-feedback
**Done:** Run panel reordered (time, finish clock time, then a class | frames-to-run table), Estimate, Preview and
job status lines in the same order; Cancel ends the worker at once, also while pausing (D34a), and a pausing job
says what it waits for; `aten::_assert_async` goes to the job log only (D36b). Self-tested with real SAM on the
scratch folder: Pause, then Cancel while pausing ended the job 3 s after the click; frame 9 was not written.
**Broke/surprised us:** the first try was paused by the memory guard (1.8 GB free) while the test suite ran
alongside, which is the guard working; a quiet second try passed. "pausing — pausing: …" read twice; reworded.
**Next:** QE on the PR, then Nick's S4 re-test.

## 2026-11-25 — Sprint S4 — Bites 1a–5b (builder)
**Branch:** sprint/4-jobctl-core, 4-sam-jobs, 4-extract-jobs, 4-estimate, 4-limits, 4-schedule, 4-awake (PRs 32–37, 39)
**Done:** job control as `app/jobctl/` (one queue, records in `<data_dir>/jobs/`, workers in their own session with
`pause` on stdin, interrupted after a quit, one app per data folder); SAM, extraction and playback copies on it
(`app/jobs.py` removed; Stop became Pause); Jobs page and dashboard "Unfinished job"; Estimate (1 + 5 frames) and
rolling remaining time; limits (device, threads, nice, pause, memory guard on psutil available, measured: D36a
2 GB) and the CPU-fallback warning; Start at / Stop by with Start now and missed; keep awake, per-job summary,
morning summary, log page. Each bite self-tested in the browser, most with real SAM on the scratch data folder.
QE found real races: a quick restart + Resume ran an item twice (fixed: worker PID kept in the row, the queue
waits for it); an Apply while paused was undone on Resume; Resume dropped a future Stop by.
**Broke/surprised us:** my scripted TRACKER edit once matched S2's "Bite 2" and pasted 117 lines (fixed; edits are
now scoped to the S4 section). `vm_stat` "Pages free" reaches 0 GB in ordinary SAM runs; psutil available does not.
A test raced the runner (files ready before the job row said done); fixed. The laptop was on battery during the
keep-awake self-test: the warning showed, and macOS listed the process's idle and system sleep assertions.
**Next:** QE on PRs 37 and 39; then the lead hands Nick the S4 test build (TRACKER Demo).

## 2026-11-25 — Sprint S4 — Plan (planning lead)
**Branch:** sprint/4-plan (docs only)
**Done:** S4 (job control) planned under CLAUDE.md rule 7: five bites in TRACKER S4, DESIGN §Planned for S4, Lead
calls D32–D38 (module `app/jobctl/` with one queue and records in `<data_dir>/jobs/`; worker protocol with `pause`
on stdin and progress from the artifacts; pause / resume / interrupted jobs; estimate 1 + 5 frames; limits and a
measured memory guard; schedule; keep awake). SPEC Q10 points at D36.
**Broke/surprised us:** a hook on this laptop blocks shell commands that name the macOS keep-awake tool: builders
test keep awake through the app, and tests use a fake command (DESIGN §Planned for S4).
**Next:** the S4 builder starts bite 1a.

## 2026-11-25 — Sprint S3 — Demo passed
**Branch:** sprint/3-demo-passed (docs only)
**Done:** Nick ran the S3 test build on `main` at `49e1db8` and wrote "S3 passes" in the planning lead's chat
(5:03 PM). Bites 1–6 merged: PRs 20, 21, 22, 24, 27, 28; docs PRs 19, 23, 25, 26. Lead calls D22–D29, D31; Nick's
class form D30 replaced D25. S3 marked `[x]`; bites 1–3 ticked with their PRs.
**Broke/surprised us:** kelp prompts are weak (n = 6 frames; shoots come in pieces): re-check on more footage
later (icebox). Whole-frame SAM on a 5.3K frame peaked at 26.7 GB Apple GPU memory (input for S4's memory guard).
Two builders were briefly started for bites 5–6; the duplicate was archived before it changed anything.
**Next:** S4 (job control), planned by the lead.

## 2026-11-25 — Sprint S3 — Bite 6: class editor form
**Branch:** sprint/3-class-form (from sprint/3-viewer, bite 5)
**Done:** the YAML text box is replaced by a form: rows with name, kind, prompt chips, group and "pixels
count as" drop-downs, ↑ ↓, Remove, "+ Add class" with the next free id, and the SAM settings as labelled
number fields. Every change is checked on the server with the same checks as classes.yaml; Save is off
while there is a problem. A fuzz test (1,200 random form inputs) shows no input saves a file that fails
the checks, and a refused save leaves the file untouched. 168 tests.
**Broke/surprised us:** leaving a text box redrew the table and dropped the click on the next box; the
redraw now waits for the focus to move and gives it back.
**Next:** QE on the bite 5 and 6 PRs.

## 2026-11-25 — Sprint S3 — Bite 5: large viewer and class legend on the Labeling page
**Branch:** sprint/3-viewer (PR 27; built on bite 4's branch, `main` merged in after PR 24's squash)
**Done:** clicking a preview frame opens S2's viewer on the page (arrows, Esc, zoom, pan) with the masks;
a class legend (toggle, "only", final class map) drives the tiles and the viewer together; objects boxed,
areas not, and the legend says so. Masks are decoded inside their box only and cached: on a synthetic 5.3K
frame with 150 masks a toggle went from 1.9 s to 0.01–0.05 s (0.5 s the first time). 161 tests.
**Broke/surprised us:** a class toggle in the viewer reset the zoom; `viewerShow(i, true)` now keeps it.
**Next:** QE on the bite 5 PR; then bite 6 (class editor form).

## 2026-11-25 — Sprint S3 — Bite 4: Run, Stop, re-run only what changed, overlap rules
**Branch:** sprint/3-run (PR 24)
**Done:** Run on every kept frame with Stop and carry-on; the page lists the classes that will run and those
unchanged; `app/sam/overlap.py` (D22) with a "final class map" toggle. Self-test with real SAM on 8 4K frames:
Stop at frame 2, Run carried on; a new class ran alone, and the other classes' results stayed unchanged.
**Broke/surprised us:** the first estimate (preview s/frame × the share of units) said 1 min for one new class;
it took 3 min 43 s, because reading, tiling and image encoding cost about 21 s per frame however few units run.
The estimate is now a fixed cost per frame plus a per-unit cost.
QE fixes (PR 24 FAIL at `77b11bd`): class ids capped at 254 (the class map is one byte, 255 = untagged);
a frame with an unreadable `sam_raw` file is skipped and named instead of stopping Run; Stop only shows during a
Run; D31 in `overlap.py` (a reviewed mask beats an unreviewed one at IoU > 0.5).
Nick's feedback on the Labeling page (viewer, mask legend, class editor) is captured separately.
**Next:** QE on PR 24; then Nick's feedback items.

## 2026-11-25 — Sprint S3 — Bite 3 bench: kelp prompts, D10 re-check, minimum mask area
**Branch:** sprint/3-preview (main merged in after PR 21's squash, `c24dba4`)
**Done:** kelp prompts picked from the overlays of 6 restoration frames: kelp shoot "young kelp", kelp area
"kelp" ("kelp stipe" found nothing; "kelp forest" covered open water). D10 re-checked on 9 urchin frames from
5 videos (3 sites): tiles keep 3–4× more urchins on dense 4K frames, the same on close 5.3K frames, at 4–8× the
time; D10 holds. D24: the smallest of 580 urchin masks is 948 px and the small ones are real, so
`min_mask_area_px` 500 (a floor for specks). Numbers in DESIGN §Bench results.
**Broke/surprised us:** whole-frame SAM on 5312 px frames peaked at 26.7 GB Apple GPU memory (tiles 16.3 GB).
Three Wreck videos are 1280×960, so they are never tiled.
**Next:** draft PR for bite 3 to QE; then bite 4 (Run, re-run, overlap rules).

## 2026-11-25 — Sprint S3 — Bites 1–2 built, bite 3 preview in progress; builder rotates
**Branch:** sprint/3-classes (PR 20, merged `373521d`), sprint/3-sam-runner (PR 21), sprint/3-preview (no PR yet)
**Done:** bite 1: `classes.yaml` with checks naming the line, Labeling page with edit / Save / Reload, upright
EXIF helper (review thumbnails fixed). Bite 2: SAM worker process, `sam_raw/<frame file>.json`, tile-edge
duplicates (D23, cut pieces unioned after QE), area masks one per prompt, re-run by settings key. Bite 3 code:
Preview on 3 frames with toggles, counts, s per frame and peak memory; ran with real SAM in the browser
(65 s per 4K frame, 12.8 GB Apple GPU). 132 tests.
**Broke/surprised us:** the bench frames from 3 videos showed urchins on 2 frames only; "kelp stipe" found
nothing on 6 restoration frames. macOS `._` files appear in `sam_raw/` on the exFAT SSD.
**Next:** bite 3's kelp prompts, D10 re-check on more urchin frames, D24 minimum area, PR (handoff note in the notes folder).

## 2026-11-25 — Sprint S3 — Plan
**Branch:** sprint/3-plan (docs only)
**Done:** planning lead wrote the S3 plan (CLAUDE.md rule 7): four bites in TRACKER (classes, SAM runner and
results files, prompt preview, run + re-run + overlap rules), DESIGN §Planned for S3, Lead calls D22–D27
(SPEC Q2 overlap rules and kelp shoot; Q10 SAM defaults; classes edited in the browser; preview frames;
EXIF orientation of stills).
**Broke/surprised us:** nothing.
**Next:** QE reviews the plan; then the S3 builder starts bite 1.

## 2026-11-25 — Sprint S2 — Demo passed
**Branch:** sprint/2-demo-passed (docs only)
**Done:** Nick ran the S2 test build and wrote "the demo passes" in the builder chat (12:09 PM).
Every S2 PR is merged (last code PR: PR 15, filter rows, `4fad379`; then docs PR 17, `106f867`). S2 marked `[x]` in TRACKER.
**Broke/surprised us:** nothing new.
**Next:** S3 (SAM on single frames), planned by the lead.

## 2026-11-25 — Sprint S2 — Bites 4 (review frames) and 5 (stills + resume) merged
**Branch:** sprint/2-frames-4 (PR 14), sprint/2-frames-5 (PR 16), sprint/2-frames-45-done (docs only)
**Done:** Bite 4: `app/stages/review.py`; grid of 640 px copies with a 1–8 per-row slider, arrows,
`D` (to `rejected/`), `K`, `Z`; full-screen viewer shared with Extract (`viewer.js`); "Show dropped"
with `K` = `kept_hand` (D20); Skip and Done. QE Lows fixed: undo keeps a step whose frame Extract
changed (the hand mark survives a plan change); preview and status counts use the plan. Bite 5:
still-image folders copied byte for byte into `frames/`, counted apart, skipping clips, extraction
and filters (D21, Lead call); a truncated JPEG is counted unreadable (QE Medium); Extract shows how
far a stopped run got and carries on. 94 tests. S2 Demo run end to end in the builder's browser
on the bite 5 head (notes folder, `s2-demo-run-2026-11-25.md`).
**Broke/surprised us:** force-push is blocked in builder sessions, so branches take `main` by
merge, not rebase; PR 16 was stacked on PR 14 and retargeted after the squash merge.
S2 Demo re-run on `main` at `4fad379` (after PR 15): 37 kept + 3 blurry + 0 duplicate = 40 = live
count; D / Z / D; restart kept all four stages and the frame in `rejected/`; 6 stills in review.
**Next:** Nick's S2 test build; the checklist is `s2-test-build-checklist.md` in the notes folder.

## 2026-11-25 — Sprint S2 — Filter rows simplified (Nick's feedback on 3b)
**Branch:** sprint/2-filter-rows (from main `87310c0`)
**Done:** Nick found the filter examples too much and their labels confusing ("the sharpest
frames dropped"). Now one row per filter showing every frame it would drop, scrolling sideways; click
opens the viewer. Order (Nick's second look): most blurry and least unique (fewest bits
differing from the last kept frame) on the left. The blur row follows the setting at which each
frame drops, so a stronger blur only adds on the right (QE); the duplicate row can still gain
frames mid-row.
After PR 14 (bite 4) merged: main merged in, `kept_hand` kept (never ranked by blur, still a
duplicate anchor), and the rows load the 640 px `/framethumbs/` copies; the viewer still opens
the full-size frame. 89 tests. The kept-side and paired examples are gone. Browser
check on the scratch project at 30 % / 20: rows of 12 and 14 in order (sharpness 4 → 141,
11 → 19 bits). 82 tests.
**Broke/surprised us:** Nick's app on port 8000 was an old copy from the previous builder's
folder, so he saw no filters at first.
**Next:** QE reviews the draft PR; the lead merges.

## 2026-11-25 — Sprint S2 — Bite 3b merged; builder rotates
**Branch:** sprint/2-frames-3b-done (docs only)
**Done:** PR 9 (bite 3b) merged by the lead as `1284232`, matching QE's PASS on `5a72243`, under
the new working rules (PR 10, `36b499f`). TRACKER 3b items marked done.
**Broke/surprised us:** Nick did not run the 3b manual test before the merge; under the new rules
it moves into the S2 test build (the four filter steps in PR 9's description).
**Next:** bite 4 (review frames) by the next builder session; handoff note in the session notes.

## 2026-11-25 — Sprint S2 — Bite 3a merged; bite 3b: blur and duplicate filters
**Branch:** sprint/2-frames-3b (from main `f9f14b2`, after PR 7 merged)
**Done:** PR 7 (bite 3a) merged as `f9f14b2` after QE PASS on `5a3d96d` and Nick's word. Bite 3b
plan approved by Nick. `extract.py`: a pure filter rule (per video, blur first, then duplicates
against the last kept frame), a read-only preview with examples at each cut-off, and Apply,
which moves files between `frames/`, `dropped/blurry/` and `dropped/duplicate/` (never erasing)
and repairs a file left in the wrong folder by a stop. Extract applies the saved values at its
end and never decodes a frame already in one of those folders (QE PR 7 watch item). Extract
page: two sliders with live counts, example frames (click opens the viewer), Apply. 8 new tests
(82 in total). Browser check on the scratch copy of 44 real frames: 30 % / 20 bits gave
18 + 12 + 14 = 44 and the folders held 18 / 12 / 14 files; Extract again decoded nothing;
0 % / 0 brought all 44 back.
**Broke/surprised us:** with the default duplicate value an older test's synthetic video lost a
frame as a duplicate (its frames look alike), so that test turns the filters off. Example
frames are full-size 4K files and take a moment to appear.
QE PASS-with-nits on `845ec27`: rebased on main, decision renumbered D19 (D18 is the urchin type from PR 8), a non-integer duplicate value now gets 400.
**Next:** QE reviews the draft PR; Nick runs the 3b manual test; then bite 4 (review frames).

## 2026-11-25 — Sprint S2 — Bite 3a: QE findings fixed
**Branch:** sprint/2-frames-3 (PR 7, still a draft)
**Done:** QE PASS-with-nits at c182fa2 had three findings. (1) Frames from earlier settings stayed `kept`
  (12 kept for a plan of 3): they now move to `dropped/unplanned/` and come back if asked for again (D17a).
  (2) Rotated videos would fail every frame: decoding goes through a PPM pipe, so frames come out upright.
  (3) TRACKER 3a items marked [x]. Two tests added; 72 pass.
  Nick asked for an in-page frame viewer (like iPhoto): click a frame to see it large, arrows or < > move
  through every kept frame, Esc or Close leaves. Then full screen (browser Fullscreen API), no frame
  around the image; click zooms to actual pixels at that point, scroll / pinch zooms, drag pans, 0 fits.
  Nick did not see the changes: the browser kept old saved copies of the scripts. Script and style
  links now carry a content hash (`?v=`) and /static answers `Cache-Control: no-cache`.
  QE PASS-with-nits at 730b58e; its Low finding fixed: a stop between moving a frame and updating its
  row is repaired on the next run (test added, 74 pass). The 3b watch item is in TRACKER.
**Broke:** nothing known.
**Next:** QE verdict on the new head (new QE session not running yet); Nick's manual test; then bite 3b.

## 2026-11-24 — Sprint S2 — Bite 3a: sampling plan, extraction, counts
**Branch:** sprint/2-frames-3
**Done:** Nick said to keep working in this session (instead of rotating). `app/stages/extract.py`:
sample times inside clips (whole video if none), cap spread evenly over all samples; frames
decoded from the original with ffmpeg and saved by Pillow at JPEG quality 95 as
`<video-stem>_<path-code>_<ms>.jpg` (spaces in the stem become _); blur score and dHash stored per
frame with its video and timestamp; temp name first, one database row per frame, so a re-run only
makes missing frames. Extract page: settings, live count while typing, background job with
progress, counts, the first 12 frames. 7 new tests (70 in total). Real run on 2 SSD videos
(scratch project): 44 frames as planned, 3840 px wide, 46 MB, 20 s; blur scores 4.5–584.
**Broke/surprised us:** nothing.
**Next:** QE reviews the draft PR; Nick runs the 3a manual test; then bite 3b (filters).

## 2026-11-24 — Sprint S2 — Bite 2 merged; bite 3 planned (session rotates)
**Branch:** sprint/2-frames-3 (from main `c730e20`)
**Done:** PR 6 (bite 2b) merged as `c730e20` after QE PASS on `5e52833` and Nick's "ship it".
Bite 3 plan approved by Nick: split 3a / 3b, blur cut-off per video, dropped frames moved to
`dropped/` folders (DESIGN D17). Measured on two SSD videos (scratch output only): full-size frame
0.5 s (4K) / 0.8 s (5.3K) by seek; JPEG q95 save 0.02–0.04 s, 1.5 / 2.0 MB; blur + dHash ~0.03 s.
**Broke/surprised us:** sharpness scores differ ~5x between videos (why D17 is per video). A running
app copy serves new templates with old code after a branch switch (500s in Nick's log); restart fixes it.
**Next:** bite 3a in a fresh builder session (TEAM_RULES 9: this one covered bites 1–2).

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
From Nick's manual test: clearer refusal messages and button names (Set start here / Set end
here, hover hints); a clip's band and its row light up together when either is clicked; each row
shows the clip's first frame (made from the copy at that time, cached in `clipthumbs/`), which
changes when the start is trimmed. 63 tests.
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
