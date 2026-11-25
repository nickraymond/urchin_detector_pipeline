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
