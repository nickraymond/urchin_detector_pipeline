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
