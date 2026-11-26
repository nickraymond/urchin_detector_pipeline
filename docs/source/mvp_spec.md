# Urchin Monitoring MVP — Spec

Nov 24, 2026 · @Nick

## MVP goal

Pick the go-to-market camera (N6 vs Pi) by testing which one, fixed in one spot, reliably measures urchins, sea stars and kelp across a full day of light in turbid water. Hourly Iridium reporting is built only for the chosen camera. This doc covers the whole MVP. The first build is the labeling tool, specified from "Build 1" down.

| Question scientists ask | Hourly metric that answers it |
| --- | --- |
| Is urchin cover changing? | % cover per urchin type |
| More urchins, or one urchin close to the lens? | Urchin count + largest single object as % of frame |
| Are sea stars returning? | Sea star count |
| How do urchin types change with season and temperature? | Count and % cover per type, joined in the cloud with buoy sensors by timestamp |
| Did the planted kelp survive, or did urchins eat it? | Kelp % cover + kelp shoot count |

**Starting classes:** purple urchin, red urchin, sea star, sunflower sea star, anemone, fish, kelp shoot, kelp, rock/substrate, unknown. Anemone and fish are labeled for later use (fish as one class for now; rockfish species can be split out in a later training round); they have no field in the hourly packet yet. Kelp is back in the starting list (2026-11-24): the restoration-site videos show kelp, so there are reference frames. Confirm the urchin types (class discovery + Reef Check) before labeling at scale. Classes live in a settings file, so adding one later is a config change. Class ids are only ever appended, never renumbered or reused. The long-term class list follows the Reef Check California manual's species list; the starting list is kept small to prove the pipeline.

**Not in the MVP:** panoptic segmentation, tracking individual animals, kelp growth rates in cm/day.

## On the camera: two small models, one hourly packet

Each hour the camera wakes, captures a few frames, keeps those that pass a quality check, runs two models, and sends one packet.

- **Detector** (boxes) for counts: purple urchin, red urchin, sea star, kelp shoot.
- **Segmenter** (a class for every pixel) for % cover: urchin per type, sea star, kelp, rock/substrate, unknown.
- **Quality gate:** skip frames that are too murky, dark or blurry; report how many frames were valid.
- **Licenses:** permissive only (Apache / MIT). Choose architectures already shown to run on the N6 NPU.
- Speed is not a constraint at one report per hour, so frames can be tiled for small urchins.

**Hourly packet, after down-select only (about 21 bytes, so one 200-byte Iridium message can carry several hours):**

| Field | Bytes |
| --- | --- |
| Timestamp + model version | 5 |
| Valid frames + quality score | 2 |
| Urchin count per type (2 types) | 4 |
| Sea star count | 1 |
| Kelp shoot count | 1 |
| % cover: each urchin type, sea star, kelp, rock, unknown | 6 |
| Largest object % of frame | 1 |
| Flags: close-up, count saturated, murky | 1 |

**Flag rules (thresholds tuned in the field):** close-up = one object fills a large share of the frame; saturated = cover keeps rising while count stops (count is then a floor); murky = too few valid frames.

## Camera bake-off

The same trained models run on each camera, side by side, looking at the same patch of seafloor. Each host has two sensors: the Pi a 12 MP IMX708 for detail and a 2 MP STARVIS for low light; the N6 a 5 MP and a 1 MP global shutter. The AE3 is a stretch goal (below): its 1 MP sensor and limited flash make it the long shot.

| Camera | Sensor | Role |
| --- | --- | --- |
| Raspberry Pi | 12 MP (IMX708) | Reference: best image quality, too power-hungry to ship |
| Raspberry Pi | 2 MP STARVIS (IMX462) | Low-light option with 1080p video; tests whether low-light sensitivity beats resolution in turbid water |
| OpenMV N6 | 5 MP rolling shutter, HDR stills | Candidate go-to-market sensor; test HDR stills vs. standard stills on high-contrast and turbid scenes |
| OpenMV N6 | 1 MP global shutter | Sharp frames despite surge and camera motion; tests whether that offsets lower resolution |
| OpenMV AE3 (stretch) | 1 MP global shutter | Lowest-power option, only if the same models deploy easily |

**One-day field test, dawn to dusk:** all cameras side by side underwater for one day, so each sees the full range of light. Compare % cover and count error against hand-labeled Pi frames from the same day, broken down by time of day; share of valid frames; and energy per inference.

**Open question:** how close the N6 must get to the Pi to win. Proposal to start: within 10 percentage points on cover and ±20% on counts.

**Stretch goal: AE3.** Try deploying the same trained models to the AE3 (Ethos-U55 NPU; models compiled with Arm's Vela tool). The test is ease, not just accuracy:

- **Pass:** the detector and segmenter, or at least the detector, convert, fit in flash and run on the AE3 in about one day of work, without retraining a special architecture for it. Then it joins the bake-off.
- **Disqualified:** the models don't fit, need unsupported layers, or need a custom AE3-only model. Record why in one line and drop the AE3 for this product.

The time box keeps the AE3 from eating the MVP schedule.

## MVP milestones

1. Build 1 (this spec): labeling tool, run on a few GoPro videos; confirm the classes.
2. Label about 300 frames; train the detector and segmenter; quantize.
3. Bench test both models on the N6 and the Pi; check accuracy loss from quantizing. Stretch: one-day AE3 deployment attempt (pass or disqualify).
4. Quality gate + on-camera metrics, logged to SD card for the bake-off. No Iridium until one camera is chosen.
5. Field bake-off: all cameras underwater for one day, dawn to dusk; decide the go-to-market camera. Then add the hourly Iridium packet to that camera only.

## Build 1: labeling tool — overview

A local app that turns a few GoPro videos into reviewed training data for both models: boxes for the detector and per-pixel class masks for the segmenter, from one SAM 3.1 pass. User: Nick, on his own laptop; no cloud.

**Sandbox limit (configurable):** at most 300 frames per project, with no limit on the number of videos, drawn from marked clips if any are marked, otherwise from whole videos.

1. **Ingest** — pick a folder or files; the app lists the GoPro videos it found.
2. **Pick clips (optional)** — scrub each video's timeline and mark the stretches worth extracting from.
3. **Extract frames** — sample stills from marked clips; drop blurry and near-duplicate frames.
4. **Review frames (optional)** — grid view; delete frames with nothing of interest.
5. **SAM 3.1 labeling** — show a time estimate, then run each class's prompts. Pausable, resource-limited, and schedulable ("start processing at 10 pm").
6. **Review labels and export** — fix classes, delete bad masks, export both formats.

Each stage resumes where it left off and can be re-run without redoing earlier stages. A dashboard shows each stage's status.

**Class settings file** (`classes.yaml`): per class, a name, its kind (`object` = countable, or `area` = cover only), one or more text prompts, and optional example boxes. Starting classes: purple urchin, red urchin, sea star, sunflower sea star, anemone, fish, kelp shoot (object); kelp, rock/substrate (area); everything unlabeled = unknown.

## Stage details

### 1. Ingest

- User picks one or more video files or a folder. Recurse into subfolders.
- Accept `.MP4` / `.MOV`. Don't list GoPro sidecar files as videos: `.LRV` (low-res proxy, used only for clip scrubbing) and `.THM` (thumbnail).
- Treat GoPro chaptered files (`GX01xxxx`, `GX02xxxx`, … with the same `xxxx`) as one recording, in chapter order.
- Show per video: duration, resolution, fps, codec, file size. GoPro files are often HEVC (H.265); confirm the decoder works and warn if not.
- Skip macOS `._` shadow files. File names may be GoPro style or ReefCheck style (`<site>-<date>-<time>-<diver>-<tag>.MP4`); the chapter rule applies only to GoPro names.
- Also accept folders of still images (JPEG / PNG), for example frames from the N6 or the Pi. They skip stages 2 and 3 and join at frame review.
- Never modify or move source videos. Read-only.

### 2. Pick clips (optional)

- Per video: a player with a timeline scrubber and a thumbnail strip (one frame every \~5 s) to scan the whole video at a glance.
- Mark useful stretches with in/out points (keys `I` / `O`) and an optional note (e.g. "dense purple urchins"). Ranges can be adjusted or deleted.
- Show marked time per video and how many frames extraction will produce from it.
- Optional **Suggest clips** button: run SAM 3.1 at low resolution on the thumbnail strip and highlight stretches with urchins; the user accepts or ignores each.
- Save clips per project in `clips.json` (video, start, end, note) so extraction is repeatable.
- If the browser can't play GoPro HEVC, scrub a low-res proxy instead (the GoPro `.LRV` file, or an H.264 proxy made with ffmpeg); frames are always extracted from the full-res original.

### 3. Extract frames

- Sampling, inside marked clips only (whole video if none are marked): one frame every N seconds (default 2 s), user-adjustable, with a live count of how many frames that will produce.
- Cap at the sandbox limit (default 300). If over, spread samples evenly across all videos.
- **Blur filter:** score each frame (variance of Laplacian); drop frames below a threshold. Show the threshold as a slider with example kept/dropped frames.
- **Near-duplicate filter:** perceptual hash; drop frames too similar to the previous kept frame (camera sitting still).
- Save full-resolution frames as PNG or high-quality JPEG (quality ≥ 95). Name: `<video-stem>_<path-code>_<timestamp-ms>.jpg`, where the path code is a short code made from the video's full path, so two videos with the same name never collide.
- The live count is the number of frames sampled before the filters. After extraction, show kept, dropped as blurry and dropped as duplicate; the three add up to the live count (or to the cap, if the cap applied).
- Record each frame's source video and timestamp so any label can be traced back to the video.

### 4. Review frames (optional)

- Thumbnail grid, with a size slider (from about 8 per row down to 1 per row; default 6) and a full-screen view (key F; arrow keys move between frames). Each thumbnail is the whole frame, scaled down, never cropped; click it to open full size.
- Keyboard first: arrow keys to move, `D` delete, `K` keep, `Z` undo.
- "Delete" = move to a `rejected/` folder, never permanently erase.
- Optional **"Pre-screen with SAM"** button: runs a fast low-resolution SAM 3 pass and flags frames with zero urchin detections, so the user reviews only the flagged ones.
- User can skip this stage entirely.

### 5. SAM 3.1 labeling

- Model: Meta SAM 3.1 (gated Hugging Face repo `facebook/sam3.1`), text-prompt mode: one prompt returns a separate mask, box and score for every match. Fallback: `facebook/sam3` behind the same interface.
- Run every prompt from `classes.yaml` on every frame; tag each mask with its class and prompt.
- **Object classes** (urchins, sea star, kelp shoot): keep each instance separate.
- **Area classes** (kelp, rock/substrate): keep the mask; expect lower quality than objects.
- **Overlap rules:** when two object prompts claim the same object (mask IoU > 0.5), keep the higher score and flag it for review. Objects sit on top of areas. Pixels no class claims = unknown.
- **Prompt preview:** before a full run, run on 3 sample frames and show the masks, so prompts and thresholds can be tuned.
- Settings: confidence threshold (default 0.4), minimum mask area, inference resolution.
- **Tiling** (on by default for frames wider than 2000 px): overlapping tiles (about 20%), results merged to full-frame coordinates, duplicates across tile edges removed.
- Each saved result records the prompt text and the settings that made it. Changing a prompt or a setting re-runs only that class. Masks Nick already reviewed are kept and marked "from older settings"; they are never silently replaced.
- Save results after each frame, including frames with zero detections. Never send images off the laptop.

### 6. Review labels and export

- Per frame, with the same zoom slider and full-screen view: image with colored masks by class; a toggle to show objects only, areas only, or the final per-pixel class map.
- Click a mask to select it. Searchable class picker to change its class; `X` deletes it; `Enter` accepts the frame.
- **Urchin type** (2026-11-25): SAM runs one prompt, "urchin", because it cannot tell purple from red. `P` / `R` tags the selected urchin purple or red; a frame cannot be accepted while it has an untagged urchin. Exports keep the two classes purple urchin and red urchin.
- **Split check:** per frame, mark whether touching urchins were separated correctly. Frames marked "no" still count for cover but are left out of detector training.
- **Fully checked flag:** per frame and class, record that the reviewer looked for that class. Training skips a class on frames never checked for it, so classes added later don't turn old frames into false "background".
- Running totals per class (objects and frames), so the user knows when a class has enough examples (target 150–200 objects per class).
- **Add a missed object:** click on it (or drag a box around it); SAM returns a mask for that spot; pick its class. The mask is saved like any other, marked as added by hand.
- No hand-editing of mask outlines in v1. Export to CVAT for fine fixes.

## Job control: estimate, pause, resource limits

Long-running stages (extraction, SAM pre-screen, SAM labeling) run as a background job in a separate worker process, so the UI stays responsive and the job survives closing the browser tab. Job control is its own module with no labeling-specific code, so later training and evaluation jobs can use the same queue, schedule, pause and log.

**Time estimate before Run**

- On "Estimate", time SAM on 5 sample frames (after a 1-frame warm-up that is not counted), at the chosen settings and resource limits.
- Show: seconds per frame, total frames, estimated total time, and estimated finish clock time if started now or at the scheduled time.
- During the run, update the remaining-time estimate from a rolling average of the last 20 frames.

**Pause, resume, cancel**

- Pause finishes the current frame, then stops and frees GPU/CPU memory (unloads the model).
- Resume reloads the model and continues from the next unprocessed frame.
- Progress is saved per frame, so a crash, laptop sleep or reboot loses at most one frame. On restart the app offers to resume any unfinished job.
- Cancel stops the job and keeps completed results.

**Resource limits (settings panel)**

- Device: auto-detect Apple Silicon GPU (MPS), NVIDIA GPU (CUDA) or CPU; user can override. Fall back to CPU with a warning if an operation is unsupported on the GPU.
- CPU threads cap (default: half the cores).
- Run at low OS priority (`nice`) so the laptop stays usable.
- Optional pause between frames (default 0 s) to cut sustained heat and fan noise.
- Memory guard: if free memory drops below a threshold, pause automatically and say why.
- Batch size 1 by default.

**Overnight mode**

- Basic scheduler: a "Start at" time picker (e.g. 22:00) on every long job (frame extraction, SAM pre-screen, SAM labeling), plus an optional "stop by" time (e.g. 07:00) that auto-pauses. A queued job shows its start time and can be cancelled or started now. The schedule survives an app restart.
- A **Keep laptop awake** toggle in the UI (it runs macOS `caffeinate`), offered whenever a long or scheduled job is started and released when the job ends. Next to it, a reminder: keep the laptop plugged in with the lid open.
- Write a log file and show a morning summary: frames done, urchins found, errors, time taken.

## Outputs and folder layout

One set of reviewed labels exports in two formats, one per model. Both use the full-size frames; there are no per-animal cut-outs.

| Output | What it is | Used for |
| --- | --- | --- |
| `export/images/` | Accepted full-resolution frames | Input for both models |
| `export/detector_coco.json` | COCO: per object a box, outline and class (object classes only; frames failing the split check left out) | Detector training |
| `export/seg_masks/` | One PNG per frame, each pixel = a class id, plus `classes.json` mapping ids to names and colors | Segmenter training |
| `export/splits/` | Train / valid / test, about 70 / 20 / 10, split by site and dive date (read from the folder and file name) so near-identical frames don't leak between sets | Fair evaluation |
| `export/frame_metrics.csv` | Per frame: source video, timestamp, count and % cover per class, largest object % | Ground truth for the hourly packet fields; sanity checks |

Project folder:

```
projects/<name>/
  project.json        settings
  classes.yaml        class list and prompts
  state.sqlite        job state, review state, fully-checked flags
  frames/             extracted frames
  rejected/           frames deleted in review
  sam_raw/            one JSON per frame: masks (RLE), boxes, scores, class, prompt
  export/             see table above
  logs/
```

Re-exporting is always safe: it rebuilds `export/` from `sam_raw/` plus the review state. Raw masks are kept, so new classes or formats never need SAM re-run on already-labeled classes.

## Build notes for the code agent

### Prior art: check first

Nick built a similar labeling pipeline earlier (RF-DETR based) and has his own labeling UI. Before writing new code, search his repos for it and reuse what fits: frame extraction, the UI, data formats. Report what was reused.

### Suggested stack

- Python 3.11+, local only, one command to start (`make run` or `uv run app`).
- Backend: FastAPI. Worker: separate process with a simple job queue (SQLite-backed). UI: plain HTML + htmx or a small React page, served at `localhost`.
- Video: `ffmpeg` / PyAV. Image filters: OpenCV, `imagehash`.
- SAM 3.1: run through Meta's official code (GitHub repo `facebookresearch/sam3`) with the `facebook/sam3.1` checkpoint. That checkpoint has no Transformers integration, so don't use the Transformers SAM 3 classes. Download with `huggingface_hub` after the user runs `hf auth login`; never store the token in the project. If 3.1 access isn't granted yet, fall back to `facebook/sam3` behind the same interface. License: Meta's SAM License (royalty-free, commercial use allowed); include its text in the repo's `THIRD_PARTY_LICENSES` file.
- Masks stored as RLE; polygons made only at export (`pycocotools`).

### First step: SAM test on Nick's laptop

Before any tool code, test SAM 3.1 on Nick's M1 Max (32 GB): does it load and run on the Apple GPU (MPS) or on CPU, how many seconds per 4K frame, and how much memory. The official readme lists a CUDA (NVIDIA) GPU as a prerequisite and does not mention Apple chips, so this is unverified. If it does not run, stop and choose between the SAM 3 fallback, CPU only, or another machine.

### Public repo

The GitHub repo is public. No real frames, video stills, people's names or tokens in git, PRs, issues or test fixtures. Tests use synthetic images.

### Out of scope for Build 1

- Training the models, quantizing, camera firmware, the hourly packet (later milestones)
- Hand-editing mask outlines (use CVAT); color correction of frames (later)
- Panoptic export, tracking individual animals across frames
- Multi-user, cloud, login

### Acceptance criteria

- [x] Point at a folder with 2 GoPro videos; frames extract with blur and duplicate filtering; kept + dropped counts add up to the preview count — *S2 demo passed (Nick, 2026-11-25, TRACKER S2)*
- [x] Frame review deletes to `rejected/` and undo works — *S2 demo passed (Nick, 2026-11-25)*
- [x] Adding a class to `classes.yaml` and re-running labels only that class on existing frames — *S3 demo passed (Nick, 2026-11-25)*
- [x] Estimate shows seconds per frame and finish time before Run — *S4 demo passed (Nick, 2026-11-25, TRACKER S4)*
- [x] Pause mid-run, quit, restart, resume: no frame processed twice or skipped — *S4 demo passed (Nick, 2026-11-25)*
- [x] Thread cap and low priority work: laptop stays usable during a run — *S4 demo passed (Nick, 2026-11-25)*
- [x] Scheduled start runs unattended and writes a morning summary — *S4 demo passed (Nick, 2026-11-25)*
- [x] `detector_coco.json` passes a `pycocotools` load check; `seg_masks/` PNGs match image sizes and use only ids in `classes.json` — *the Export page's checks (S5 bites 2, 6a) PASS on an export of 16, then 3, reviewed frames from one ReefCheck video (builder, 2026-11-26); Nick re-checks in the S5 test build*
- [ ] `frame_metrics.csv` counts and cover match the reviewed labels on 5 spot-checked frames — *open: Nick's spot check in the S5 test build (the Export page's metrics check compares every frame's cover with its mask and counts with COCO)*
- [x] README with setup, SAM 3.1 access steps and a 10-minute walkthrough — *README (S5 bite 6b); the steps are for SAM 3, the model in use (D10)*

### Build 1 milestones

1. Ingest + clip picker + frame extraction + frame review (no SAM)
2. SAM 3.1 on single frames: class settings file, prompt preview, tiling, overlap rules
3. Job control: estimate, pause/resume, limits, overnight mode
4. Label review + both exports + frame metrics; acceptance test on Nick's GoPro videos
