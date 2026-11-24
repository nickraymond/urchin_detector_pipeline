# DESIGN.md — Architecture & Decisions (as-built)

*What it did / how it's shaped. Agents append; never silently rewrite history.*
*Last updated: 2026-11-24*

## System topology

S1 (a test, not the tool): `uv run sam-test --video <file>` → `app/sam_test/run.py` pulls 3
frames with ffmpeg into `<data_dir>/sam_test/frames/`, then starts `app/sam_test/worker.py` as a
separate process for each attempt (Apple GPU → Apple GPU with CPU fallback → CPU, stopping at
the first that works). Each worker writes `result_<attempt>.json` after every step;
`report.py` turns them into `<data_dir>/sam_test/report.html` with the overlays embedded.

## Key designs

- Meta's code is installed unmodified (pinned commit); `device_patch.py` makes the
  run-time changes it needs off-CUDA and lists every one on the report page (D8).
- One process per attempt, so a hard crash in the Apple GPU driver still leaves a report (D9).

## Decision log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-11-24 | Data and local paths live outside git: `config.local.toml` (gitignored) names the video folders and the data folder on the SSD; `config.example.toml` is the committed pattern. | SPEC hard constraints 2–3; Nick chose the SSD (chat 2026-11-24). |
| D2 | 2026-11-24 | Branch names are `sprint/<n>-<name>`; the word "slug" is not used. | Nick, chat 2026-11-24. |
| D3 | 2026-11-24 | SAM 3.1 is tested on the laptop (sprint S1) before any tool code. | The official readme lists a CUDA GPU as a prerequisite; the laptop is an M1 Max. Nick, chat 2026-11-24. |
| D4 | 2026-11-24 | Job control (queue, schedule, pause, log) is a standalone module with no labeling-specific code. | Nick wants to reuse it for nightly training and evaluation jobs. Nick, chat 2026-11-24. |
| D5 | 2026-11-24 | Kelp and kelp shoot classes are deferred; classes stay a settings-file change. | Nick has no reference images yet. Nick, chat 2026-11-24. |
| D6 | 2026-11-24 | The class list starts small (purple urchin, red urchin, sea star, sunflower sea star, anemone, fish, rock/substrate) and grows toward the Reef Check California manual's species list. Class ids are only ever appended, never renumbered or reused. | Prove the pipeline first; splitting a class later costs a re-review of that class only. Stable ids keep old exports and models readable. Nick, chat 2026-11-24. |
| D7 | 2026-11-24 | Pin PyTorch to 2.10.0 (the version in Meta's sam3 readme), Python 3.12. | torch 2.15.0 aborts on the Apple GPU on macOS 14.6 (`MPSNDArray arrayViewWithShape:strides:` unrecognized selector) on a plain Conv2d; 2.8.0–2.14.0 pass the same test (S1 test, 2026-11-24). |
| D8 | 2026-11-24 | Run Meta's SAM 3 code unmodified; apply small run-time fixes in `app/sam_test/device_patch.py`: stub the triton-only `edt` module, build two caches on the chosen device instead of `"cuda"`, replace the fused bfloat16 `_addmm_activation` with float32 linear + activation (its bfloat16 output gave NaN scores on MPS and a dtype error on CPU), skip empty point/box encodes and `pin_memory` off-CUDA. Other missing ops use PyTorch's CPU fallback; the report lists which. | Smallest change that runs; Meta's files stay untouched and upgradeable. Each fix is from a failure seen in the S1 smoke test. |
| D9 | 2026-11-24 | The SAM test runs each attempt in its own worker process. | The Apple GPU can abort the whole process (seen with torch 2.15.0); the parent must still write the report. |

## Bench / test results

*(appended by sprints)*
