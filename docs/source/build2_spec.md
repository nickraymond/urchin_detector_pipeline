# Build 2: what runs where, then train, compare and bench test — Spec

Nov 26, 2026 · draft for @Nick · adds to `docs/source/mvp_spec.md` (Build 1 is the labeling tool)

**Status: draft for Nick's merge.** Nick's answers of 2026-11-26 and 2026-11-27 (spec lead's chat) are written in and listed at the end.

## Goal

In the same local app as Build 1, go from the reviewed export to a short list of models that are
proven on the bench cameras, with the numbers to pick what to deploy. Like Roboflow or Edge Impulse,
but on the laptop: train several model types, evaluate them, compare new against old and one type
against another, shrink them to fit each camera, and test them on the real hardware.

Build 2 answers two questions, in this order (Nick, 2026-11-27):

1. **What can each camera run?** For each kind of model (detector, segmenter, instance segmenter)
   and each camera (N6, Pi Zero 2W, AE3): does a model of that kind convert, fit and run, how fast,
   with how much memory and energy. This needs no trained model and no labeled data, and the
   answer is valuable on its own. It comes first.
2. **Which trained models are best?** Train on Build 1's export, compare, convert the best ones and
   measure their accuracy on the cameras that question 1 showed can run them.

This is MVP milestones 2 and 3 ("train the detector and segmenter; quantize" and "bench test both
models on the N6 and the Pi; stretch: AE3"). Build 2 is done when Nick can, without leaving the
browser: train a detector and a segmenter on a Build 1 export, see them side by side with older
models, convert both for the N6 and the Pi, run them on the bench units, and read one comparison
page that says which model to put on which camera and why.

**Not in Build 2:** the quality gate, on-camera metrics and SD logging, the hourly packet, camera
firmware, the field bake-off (milestones 4 and 5). Cloud training. Automatic search over model
settings. Using a trained model to pre-label new frames (a good later step; icebox).

## What gets trained

| Kind | Answers | Trained from | On the cameras? |
| --- | --- | --- | --- |
| Detector (boxes) | counts per class | `detector_coco.json` boxes | Yes, where "what runs where" shows it runs |
| Segmenter (a class for every pixel) | % cover per class | `seg_masks/` + `classes.json` | Yes, where "what runs where" shows it runs |
| Instance segmenter (an outline per animal) | counts and cover from one model; separates touching urchins | `detector_coco.json` outlines | Checked on every camera in "what runs where"; trained for a camera only if it runs there; always trainable on the laptop |

The MVP spec puts two models on the camera: a detector and a per-pixel segmenter. Nick's Build 2
request adds instance segmentation. Decision (Nick, 2026-11-26): all three kinds can be trained and
compared on the laptop; the cameras get the detector and the segmenter as planned; an instance
segmenter goes to a camera only if it passes the same "converts, fits and runs in about a day" test
as the AE3.

Training honors Build 1's review flags:

- A class not marked **fully checked** on a frame is skipped for that frame (never treated as background).
- Frames with split check "no" stay out of detector and instance training; they still count for cover.
- The train / valid / test splits come from the dataset version (first taken from the export's
  `splits/`, then kept per site-and-date group); nothing is re-split by frame.
- Class ids are the export's ids. Which classes a model learns is a choice per training run.
- **Build 2 trains on sea stars and urchins only (Nick, 2026-11-27):** purple, red and black urchin,
  sea star and sunflower sea star. Kelp, kelp shoot, rock, anemone and fish stay labeled in Build 1
  and wait for a later round. For the segmenters every other pixel is one class, "other". The
  class choice stays a setting per run, so adding kelp later is not a rebuild. (Lead call: the
  sunflower sea star counts as a star fish class, and the urchin types stay separate; Nick can
  reverse either.)

## Models that are safe to use

Rule (MVP spec, SPEC hard constraint 6): on-camera models are permissive only (Apache / MIT), and
architectures should already be shown to run on the N6's NPU. Build 2 adds BSD-3-Clause to the
permissive list (torchvision's code is BSD-3; Nick's merge of this spec is the approval). The licence of the **code**, of
any **pretrained weights** we start from, and of any **outside data** are three separate checks; a
model type is offered in the app only when all three are recorded in `THIRD_PARTY_LICENSES`.

Checked on 2026-11-26 (lead = read by the spec lead at the source; helper = read by a research
helper, to be re-read by the builder before the model type is added):

| Model family | Licence (source) | Role in Build 2 |
| --- | --- | --- |
| YOLOX Nano / Tiny | Apache-2.0 (`Megvii-BaseDetection/YOLOX` LICENSE; lead) | **First detector.** Nick's earlier work compiled it for both NPUs and measured it on both boards (below). |
| RF-DETR Nano–Large (detector), RF-DETR-Seg Nano–2XLarge (instance segmenter) | Apache-2.0 (`roboflow/rf-detr` README; lead) | **Candidates in "what runs where"** at the Nano size (Nick, 2026-11-27), and the laptop reference for "how good could it get". Never compiled for the cameras so far; attention layers are expected to stay on the CPU (the earlier compile test saw even single transpose layers fall back). The table will show it. |
| RF-DETR Atto / Femto / Pico (detectors smaller than Nano; package `rfdetr[plus]`) | Roboflow Platform Model License, PML 1.0 (`roboflow/rf-detr_plus` LICENSE; lead): use needs an accepted Roboflow platform plan; the README says commercial self-hosting "requires the Enterprise add-on"; the licence allows usage tracking in the software | **Evaluation only** (Nick, 2026-11-27): candidates in "what runs where" and on the laptop, to see how good the licensed small sizes are. Their weights, trained ones included, stay under PML: never on a field unit or in a product until Nick has the licence. Shown with a "Roboflow licence" mark everywhere. Nick has accepted Roboflow's terms on his own account (2026-11-27); before any frame goes through the package the builder checks what it sends out (hard constraint 1). |
| torchvision DeepLabV3 / LR-ASPP with a MobileNet backbone | code BSD-3-Clause (helper); weights: torchvision says they "may have their own licenses" (helper) | Candidate first segmenter. NPU fit unproven: it must pass the compile check first. |
| YOLACT, SparseInst, RTMDet-Ins | MIT (YOLACT LICENSE; lead), MIT (SparseInst; helper), Apache-2.0 (`open-mmlab/mmdetection` LICENSE; lead) | Candidate small instance segmenters. NPU fit unproven. |
| NanoDet-Plus | Apache-2.0 (helper) | Second choice: its channel-shuffle layers fell back to the CPU on both NPUs in the earlier compile test. |
| **Ultralytics YOLO (v5 / v8 / v11)** | **AGPL-3.0**, code and trained models (`ultralytics/ultralytics` LICENSE; lead; ultralytics.com/license; helper) | **Not allowed**: permissive only stays (Nick, 2026-11-26). Not in the app at all, not even as a laptop comparison. |
| ST model zoo's own models (ST YOLOX, ST YOLO LC, …) | ST licence SLA0044: may run only on ST chips (helper) | Not used: the same model must also run on the Pi and the AE3. |
| YOLO-NAS weights | non-commercial (helper) | Not allowed. |
| Edge Impulse FOMO | no licence statement found for use outside Edge Impulse (helper) | Not used until verified. Nick's own FOMO-style net (plain conv, trained from scratch) is his code and is fine. |

Different cameras may need different models or sizes. The app treats "model type × size × input
size × tiling" as one recipe, and the comparison page shows which recipes fit which camera.

What Nick's earlier work already measured (repo `ADIN_SPI_OpenMV`, `ml/yolox_urchin/STAGE1.md` and
`ml/compile_gate_report.md`, read 2026-11-26; single class "urchin", public urchin datasets, not
Build 1 data):

- YOLOX-Nano and -Tiny, with the first layer swapped for a plain stride-2 conv, compile for both
  NPUs with no layer left on the CPU: AE3 = one `ethos-u` block; N6 = 117 hardware + 2 hybrid + 0
  software steps of 119 (the trained Nano export; the untrained one was 116 + 2 + 0 of 118; Tiny 95 + 2 + 0).
- Measured on the boards (2026-08-24, 300-run windows): N6 10.6 ms (Nano) and 31.2 ms (Tiny);
  AE3 26.4 ms and 58.4 ms. Energy per inference, gross: AE3 6.7 mJ (Nano), N6 11.5 mJ (Nano).
- Tiny is too big to load from the AE3's file flash (loading copies it into memory: 4.97 MB against
  about 4.09 MB free); it runs from the read-only model area instead.
- On the screen-in-loop rig (that repo's `docs/TRACKER.md`, 2026-08-25 entry), a whole frame shrunk to
  256 px found almost nothing; tiles at native
  pixels were the only mode that worked. So tiling is part of the recipe from the start.
- Still owed there: accuracy lost by 8-bit conversion at the deployed size was never measured.
  Build 2 measures it for every model.

Reuse first (repo CLAUDE.md, value 2): the YOLOX trainer, the export script, the on-board decode
and the compile scripts from that work are the starting point; the builder reports what was reused.

## First result: what runs where

One page in the app, a table of model kind × camera, filled in before any model is trained:

| | OpenMV N6 | Pi Zero 2W | OpenMV AE3 |
| --- | --- | --- | --- |
| Detector | known: YOLOX Nano / Tiny run (earlier work) | not measured yet | known: Nano runs; Tiny from the model area only |
| Segmenter | unknown | unknown | unknown |
| Instance segmenter | unknown | unknown | unknown |

- **How a cell is filled.** Take each candidate architecture from the safe list, untrained (speed,
  size and layer placement do not depend on what the model has learned), at the input sizes of
  interest. Convert it to 8-bit, put it through the camera's compiler, then run it on the board.
- **Each cell shows, per candidate:** converts yes / no; layers on the NPU / hybrid / on the CPU
  (for the Pi: runs on the CPU); file size and memory against the board's limits; ms per inference;
  energy per inference where a power meter is wired (see stage 12); and the reason when it fails, layer
  by layer. A cell's headline is its best candidate: "runs", "runs, partly on the CPU" or "does
  not run", with the numbers.
- **Two steps.** On the laptop first: convert and compile (no hardware). Then on the bench units:
  load and time (the team's test session runs it, with a test card).
- **The AE3 is a full column here**, still under the MVP spec's time box: about one working day in
  total across the three kinds, compile and timing together; what does not convert in that time is recorded as "does not run"
  with its reason.
- **More than one model at once (Nick, 2026-11-27).** The MVP puts two models on a camera (detector
  + segmenter), so the page also answers, per camera: can it hold and run several models, or only
  one kind at a time? For each pair (and all three) of the best candidates it shows:
  - **stored together:** do the files fit side by side in the model area (N6, AE3) or in storage and
    memory (Pi), and how much room is left;
  - **loaded together:** can both sit in memory at once, with the peak memory measured;
  - **run in one wake-up:** if not together, can the camera load, run and unload one after the
    other, and how long and how much energy the whole sequence takes.
  The headline per camera is one of: "both at once", "one after the other" or "one kind only".
  Known from the earlier work: one N6 model image held two YOLOX detectors plus the vendor's
  models at 75.7% of the 24 MiB model area, and both fit on the AE3's 8 MB file flash with 1.1 MB
  free (`ml/yolox_urchin/STAGE1.md`); nothing in those notes measures two models in memory at once.
- **What it decides.** Only kinds and architectures that run on a camera are trained for that
  camera. A kind that runs on no camera is still trainable as a laptop reference.
- Numbers already measured in the earlier work are shown as "earlier work" with their date until
  they are re-measured through the app.

## Stages (added to the project page after Export)

The stages below are listed in the order of the data flow. The order of work is in the sprint
ladder at the end: stage 11's untrained fit check and stage 12's timing run come first.

### 7. Datasets

- **Freeze a dataset version** from the current export: a named, numbered snapshot (frames, labels,
  splits, class list, counts per class and split, a hash). Models are always trained on a version,
  never on "whatever the export is today", so results can be repeated.
- Shows what training will see: objects and frames per class and split, classes below the 150–200
  object target, frames left out by the split check, classes not fully checked.
- **One fixed test set for comparing.** A new version keeps every site-and-date group in the split
  it had in the version before (the export may re-deal groups on each run; the dataset version does
  not), and new groups are dealt to keep about 70 / 20 / 10. When a new version adds test frames,
  the page offers to re-evaluate older models on them, so old and new are compared on the same
  frames. The compare page flags any model that saw a test frame in training.
- **Outside data (Nick, 2026-11-26):** Nick's earlier urchin work already gathered public urchin
  datasets with their licences recorded: the data in `~/nereus_ml/datasets/`, the licence texts and
  counts in `ADIN_SPI_OpenMV/ml/urchin_data/` (`licenses/`, `manifests/`), the verdict per source in
  that repo's `docs/urchin_datasets.md` (read 2026-11-26). A dataset version may add any of these
  sources whose recorded verdict allows commercial use (for example Urchinbot, CC-BY-4.0: 9,872
  images, 44,268 boxes), each listed with its licence. Sources marked no-go or benchmark-only there
  (FathomNet, the NOAA YOLO weights) stay out. They hold boxes only and mostly one class, "urchin",
  so they are for pre-training the detector only (their boxes carry no Build 1 class id); the
  segmenters learn from Build 1's export. The DUO set stays out until Nick decides: its verdict
  there is "research-only fenced until Nick's license call".
- **Starting weights:** from scratch, from pre-training on the allowed sources above, or from
  published weights whose licence is recorded. Nick's earlier YOLOX stage-1 weights were trained on
  Urchinbot + DUO + RF100, so they wait for the DUO answer (open question); until then they are
  not a starting point for a camera model.
- Outside data is used for training only. The test split is always Nick's own reviewed frames.

### 8. Train

- Pick a dataset version, a kind (detector / segmenter / instance segmenter), a model type and size
  from the checked list, the classes (default: the urchin types and the two sea star classes), and
  the recipe: input size, tiling, epochs, augmentation
  preset (including blur and haze for turbid water). Sensible defaults; a short "why" next to each.
- Training is a job in Build 1's job module: **Estimate** (time a few steps first), Run, Pause /
  Resume from the last checkpoint, Cancel, Start at / Stop by, Keep laptop awake, resource limits,
  log and morning summary. One job at a time across the app.
- Live page: loss and validation curves, a few validation frames with predictions drawn, time left.
- Every run records what made it: dataset version, recipe, code version, random seed, library
  versions, device. Same inputs → a run that can be repeated.
- Device: the Apple GPU, with CPU fallback and a warning, as in Build 1.

### 9. Evaluate

Every trained model is scored on the test split of its dataset version, with the tiling the camera
will use. The numbers follow the questions the MVP asks, not only the usual ML scores:

- **Counts:** count error per class per frame; share of frames within ±20% (the bake-off's proposed bar).
- **Cover:** % cover error per class in percentage points; share of frames within 10 points.
- Largest object % error (the close-up flag depends on it).
- Standard scores: precision, recall and mAP50 per class for boxes and outlines; IoU per class for
  the segmenter; a breakdown for small objects.
- Per frame: the image with ground truth and predictions drawn, sortable by worst error, in the same
  large viewer as the rest of the app.
- With few test frames the page says how far the numbers can be trusted (number of frames and
  objects behind each figure). It never ranks two models on a difference the test set cannot support.

### 10. Models and compare

- **Model list:** every trained model with its kind, type, size, dataset version, date, key scores,
  file size and status (trained → converted → bench tested → picked). Models are never erased from
  the app; "archive" hides them.
- **Compare:** pick two or more models (new vs old, one type vs another, float vs 8-bit, laptop vs
  on-camera). One table, one chart per metric, and the same test frames side by side with each
  model's predictions. Differences in dataset version or test set are called out, not hidden.
- Notes per model, in Nick's words ("first run with black urchin").

### 11. Fit to the cameras

For a chosen model and each target, one job converts it and reports whether it fits. The same job
fills the "what runs where" table with untrained candidates, and later converts trained models:

| Target | Route | Report |
| --- | --- | --- |
| OpenMV N6 (STM32N6 NPU) | 8-bit model → ST Edge AI Core (`stedgeai`) → model image for the camera's read-only model area | layers on the NPU / hybrid / on the CPU; file size against the 24 MB model area |
| Raspberry Pi Zero 2W (CPU, 512 MB) | 8-bit model run by a small runtime | file size, memory needed |
| OpenMV AE3 (Ethos-U55), time-boxed | 8-bit model → Arm Vela | layers on the NPU / on the CPU; file and memory size against the board's limits |

- **8-bit conversion** uses a calibration set drawn from the training split (never the test split).
  For the untrained fit check, calibration frames only set value ranges; any training frames do.
- **Accuracy after conversion is measured on the laptop first:** the 8-bit model is evaluated like any
  other model (stage 9), so "what did shrinking cost" is one line on the compare page.
- A model that leaves layers on the CPU, or does not fit, shows why, layer by layer.
- **AE3 time box (MVP spec):** about one working day in total for getting models onto the AE3: S6's
  compile and S7's timing together. Pass = the detector and segmenter, or at least the detector,
  convert, fit and run; else disqualify with a one-line reason. Trained models go to the AE3 in S11
  only if it passed.
- Toolchain facts, checked 2026-11-26:
  - ST Edge AI Core is ST's own tool under ST's licence (not open source); current 4.1.0; it takes
    8-bit TFLite or ONNX (QDQ) models, "only full integer static quantization", and unsupported
    layers "fall back to floating-point kernels"; it runs on Apple-silicon macOS 14 or newer (ST's
    embedded docs, operator-support and setup pages; lead). The earlier work used 4.0.0 inside a
    Linux container through OpenMV's `modelc.py`.
  - Arm Vela is Apache-2.0, current 5.2.0 on PyPI, and lists macOS (lead). The earlier work used 5.0.0.
  - The OpenMV docs ship box post-processing for several YOLO formats and FOMO, and none for
    segmentation (helper): segmenter output must be decoded by our own on-camera code.
  - Pi runtime: **open.** Candidates with 64-bit Arm builds on PyPI: LiteRT (`ai-edge-litert`,
    Apache), ONNX Runtime (MIT), ncnn (BSD-3) (helper). Nick's repos hold no inference code for the
    Pi yet. Picked by measuring on the bench Pi.
  - PyTorch → 8-bit TFLite: the earlier work used `onnx2tf` (MIT); its author now points users to
    Google's LiteRT Torch, which calls itself Beta (helper). The builder keeps the known-good route
    and changes one thing at a time.

### 12. Bench test (hardware in the loop)

Two uses of the same runner. **Timing run** (first, for "what runs where"): load a converted model,
run it a fixed number of times on a few frames, read back ms per inference, memory and energy; and, for several models on one camera, load them
together (peak memory) or one after the other in one wake-up (time for the whole sequence).
**Accuracy run** (later, for trained models): the same test frames go through each converted model
on the real boards; the app collects what came back and scores it exactly as in stage 9.

- **Method A, frames from file (first):** the app sends the test frames to the board, the board runs
  the model on them and returns the raw outputs, time per inference and memory used. Every camera
  sees identical pixels, so differences are the model and the chip, not the lens.
- **Method B, screen in the loop (reuse, later):** the camera photographs frames shown on a screen
  (Nick's existing rig). It adds lens and sensor effects. Not needed to choose a model; the field
  bake-off answers the sensor question.
- **Measured per model × camera:** the same count / cover / mAP numbers as on the laptop, the gap to
  the laptop's 8-bit result (should be about zero: a gap means a runtime bug), ms per inference,
  peak memory, and energy per inference: "not measured" without a meter; with a meter, numbers marked
  "not validated" until its known-load check has passed.
- **Power is an add-on metric (Nick, 2026-11-27).** Nick will add power monitoring to the Pi 5 so it
  measures the Pi Zero 2W, the N6 and the AE3 while a model runs: power over the run, and energy per
  inference. The software does not wait for it: every run works without a meter and shows "not
  measured"; when a meter is listed for a unit in the settings file, the same run also records its
  power log and the page adds the power columns and a power-over-time chart per model. A meter's
  numbers are shown as "not validated" until a known-load check has passed. Earlier work logged an
  INA3221 at 10 Hz, one channel each for the AE3 and the N6 (`ADIN_SPI_OpenMV/ml/yolox_urchin/STAGE1.md`).
- **Test card first (team rule):** before any bench run the app shows the card (question, units,
  number of frames and repeats, metric, pass bar, stop rules); it is saved with the run. No
  recommendation from a single run.
- **Bench units only.** The app talks only to units listed as bench units in the settings file;
  field units are refused by name.
- **Rig backup first (Nick, 2026-11-27).** Before anything on the rig is changed, the test session
  takes a full backup and proves it can be put back: an image of the Pi's whole SD card, copied over
  SSH with the rig's services stopped; from each camera, over USB, its file flash, its model area
  and its firmware version; a checksum for every piece; and written restore steps. The backup is
  checked (the Pi image mounts and its files compare with the live ones; the cameras' read-backs
  match their checksums) before the first change. It lives in the laptop's data folder, never in
  git. "Restore the rig" puts everything back to that point; it is run at the end of Build 2's
  bench work or whenever Nick asks.
- **After the backup, the team has full control of the rig** (Nick, 2026-11-27: "Anything, restore
  after"): packages on the Pi, model files, the cameras' model areas, and camera firmware if a
  model route needs it. Each change is written to the rig's log so the restore steps stay true.
  On the N6, models load only by writing the whole model area over USB, so the vendor's models are
  kept in every image written (earlier work).
- **Pick:** on the compare page, mark one detector and one segmenter per camera as "picked for the
  bake-off"; the app writes a one-page report with the numbers behind the pick.

## Outputs and folder layout

```
projects/<name>/
  datasets/<version>/     frozen snapshot: manifest, splits, counts, hash (frames by reference)
  models/<model id>/      recipe, checkpoints, logs, eval results, converted files per target
  bench/<run id>/         test card, raw outputs from the board, timing, power, scores
```

Model weights, datasets and bench results live in the data folder, never in git.

## Hard constraints: what changes

- **Images never leave the laptop** stays, with one exception (Nick, 2026-11-26): test frames may
  be sent to Nick's own bench units for the bench test, over USB, the home network or Tailscale.
  Never to a cloud service.
- **Licences:** as above: permissive only for anything deployed; RF-DETR's PML sizes for evaluation
  only; nothing new without all three checks.
- **Hardware:** bench units only: the two Nick named on 2026-11-26, a Pi Zero 2W rig with the
  IMX708, an N6 and the AE3, and a Pi 5 with the screen rig. Their names and addresses live in
  `config.local.toml`, not in git. Nick sets the Pi Zero 2W rig up and hands it to one team test
  session, which owns it from then on: it takes the backup, runs the test cards the lead writes and
  QE reviews, and restores the rig. Builders and QE test against a fake board and never touch the
  hardware. Nick is asked only for physical actions (plug, power, wiring). A test card before every
  run. Field units are never a target.
- Everything else in SPEC's hard constraints stays as it is, including the browser-only rule and
  the public-repo rule (no real frames, names, unit addresses or serial numbers in git).

## Nick's answers (2026-11-26, spec lead's chat)

1. **Instance segmentation:** laptop first; on a camera only if it converts easily.
2. **Licences:** permissive only. No Ultralytics, ST-only, YOLO-NAS or FOMO models in the app. Start with YOLOX.
3. **Frames to bench units:** allowed, over any link, to his own bench units; never to a cloud.
4. **Bench units:** two of his rigs: the Pi Zero 2W rig (IMX708, N6, AE3) and the Pi 5 screen rig.
5. **Bench method:** frames from file first; the screen rig stays an option.
6. **Outside data and starting weights:** use the urchin data from his earlier detector work, which
   is already on the laptop with its licences recorded ("Look for my old work training the DETR-TR
   transformer for urchin detection and there are tons of images with the correct license I've
   captured already").
7. **Bench owner:** Nick; he runs bench tests from the app. *Replaced by answer 13.*
8. **Order (2026-11-27):** "it's actually more important to understand what hardware can run each
   type of model, detector vs segmenter vs instance segmenter. That in it-self will be valuable to
   know, then the bake later with results." So "what runs where" comes first, for all three kinds
   and all three cameras. This widens answer 1: the instance segmenter is checked on every camera
   from the start; whether it is then trained for a camera depends on that result.
9. **RF-DETR small sizes (2026-11-27):** "lets open this up to the nano and smaller versions of DETR-RT
   so we can see how good their licensed model is at smaller sizes. If it's the only way to get
   segmenter or instance segmenter on OpenMV hardware for now then that is a finding and I will
   evalaute paying for their use license." So RF-DETR Nano and the sizes below it are candidates.
   Found while writing this in: the sizes below Nano (Atto, Femto, Pico) are detectors only and
   are the ones under Roboflow's PML 1.0 licence; RF-DETR's segmentation models, Nano included, are
   Apache-2.0. This narrows answer 2: permissive only for anything
   deployed; the PML sizes are allowed for evaluation.
10. **Several models at once (2026-11-27):** "also helpful to understand if multiple models can be
   installed on the hardware at once, of if there are only enough resources for one type at a
   time". Added to "what runs where".
11. **Roboflow terms (2026-11-27):** "I have a robo flow account and agreed to their use terms so we
    are good to go for testing". The PML sizes can be tested now; paying comes only if they work
    and he wants to ship them (where payment starts: see open questions).
12. **Power (2026-11-27):** "For HIL testing I want to add power monitoring to the Pi 5 so it can
    monitor the power on Pi Zero, N6 and Ae3 so we get power as a function of model run also. But
    don't let any of that block the software buildout". Power is an optional metric on bench runs;
    Nick adds the meter to the rig himself, later. Also: "we are not emailing roboflow".
13. **Leave Nick out of the build and test cycle (2026-11-27):** "I want to be left out of this build
    and test cycle as much as possible - I will setup nereus with the zero and N6 and AE3 (I need
    you to make an image of everything on that pi and the cameras so that you can reinstall and get
    the whole test rig back up to the same exacts spot it was prior to this new testing". Then, to
    four questions: the rig is the Pi Zero 2W rig ("I will add Pi5 later to control the LCD screen
    and run power monitoring once we do HIL testing"); one team test session owns it and runs the
    bench tests; the team may do "Anything, restore after"; the Pi is imaged over SSH, no hands.
14. **Classes for training (2026-11-27):** "For trainiung lets just do star fish and urchins".

## Open questions

- The PML sizes: Nick has a Roboflow account and has agreed to its terms (answer 11), so they can
  be tested. Still to do before any frame goes through `rfdetr[plus]`: the builder shows what the
  package sends out (PML 1.0 section 4 allows usage tracking; hard constraint 1 wins). Where payment
  starts (helper, Roboflow's licensing and pricing pages, read 2026-11-27): commercial use on your
  own hardware needs their Enterprise plan with the self-hosted weights add-on; no page read says
  whether a company's feasibility test on its own bench counts as commercial. Nick,
  2026-11-27: no email to Roboflow; testing goes ahead on his account's terms.
- **For Nick:** may weights trained on the DUO set be the starting point for a camera model? Its
  licence page declares CC BY 4.0 but its origin is unclear (his earlier notes). REC: no; pre-train
  on Urchinbot and the other cleared sources only.
- **For Nick:** ST Edge AI Core is ST's own tool under ST's licence terms, which must be accepted
  to install it (he used version 4.0.0 before, through OpenMV's tooling). REC: accept for 4.1.0 on
  the laptop now: S6 starts with it; the builder reads the terms first and reports anything unusual.
- Which runtime on the Pi Zero 2W (measured in S7), and whether the bench Pi runs a 64-bit system.
- Which segmenter architecture places fully on the N6's NPU (compile check in S6).
- The licence of each set of published starting weights (checked per model type before it is offered).
- An older test card says nothing may run on the Pi Zero 2W rig, and one older skill calls it a
  field camera rig. Nick's answers 4 and 13 name it as the bench unit for Build 2; the backup step
  first lists what is running on it, and the first test card names what must be stopped.
- Power: which meter and wiring Nick adds to the Pi 5 (answer 12), and a known-load check of its
  readings. Without a meter energy shows as "not measured"; with one, as "not validated" until the check.

## Sprint ladder (detail in TRACKER)

- **S6** What runs where, on the laptop: toolchains, candidate list with licences, convert and
  compile all three kinds for all three cameras; the table's first version.
- **S7** What runs where, on the boards: the bench runner, timing runs on the N6, the Pi and the
  AE3; the table complete.
- **S8** Dataset versions + train the first detector as a job.
- **S9** Evaluate, model list, compare.
- **S10** Segmenter and instance segmenter through the same train / evaluate / compare (camera
  models only from architectures that passed S6–S7).
- **S11** Trained models on the cameras: 8-bit accuracy on the laptop, accuracy runs on the
  boards, pick report.
