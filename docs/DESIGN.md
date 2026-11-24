# DESIGN.md — Architecture & Decisions (as-built)

*What it did / how it's shaped. Agents append; never silently rewrite history.*
*Last updated: 2026-11-24*

## System topology

*(nothing built yet — filled in S1)*

## Key designs

*(filled in S1)*

## Decision log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-11-24 | Data and local paths live outside git: `config.local.toml` (gitignored) names the video folders and the data folder on the SSD; `config.example.toml` is the committed pattern. | SPEC hard constraints 2–3; Nick chose the SSD (chat 2026-11-24). |
| D2 | 2026-11-24 | Branch names are `sprint/<n>-<name>`; the word "slug" is not used. | Nick, chat 2026-11-24. |
| D3 | 2026-11-24 | SAM 3.1 is tested on the laptop (sprint S1) before any tool code. | The official readme lists a CUDA GPU as a prerequisite; the laptop is an M1 Max. Nick, chat 2026-11-24. |
| D4 | 2026-11-24 | Job control (queue, schedule, pause, log) is a standalone module with no labeling-specific code. | Nick wants to reuse it for nightly training and evaluation jobs. Nick, chat 2026-11-24. |
| D5 | 2026-11-24 | Kelp and kelp shoot classes are deferred; classes stay a settings-file change. | Nick has no reference images yet. Nick, chat 2026-11-24. |

## Bench / test results

*(appended by sprints)*
