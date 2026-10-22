# CLAUDE.md — dj-render

Rules above this file: the kernel (`~/.claude/CLAUDE.md`) and the projects SDK (`Vault/projects/CLAUDE.md`).
Vault pointer: `Vault/projects/dj-render/README.md`. Sibling library: `../dj-score` (analysis).

## What this is

The **render library**: a mix spec in, mixed audio out (time-stretch, EQ, crossfade, gain). Split out of
CyborgDJ's `cyborgdj/engine/` on 2026-10-22 so that CyborgDJ becomes a controller over two libraries:
dj-score measures, dj-render renders. An MCP layer is meant to sit on top of both.

Install into a consumer's env: `pip install -e ../dj-render`. The Rubber Band CLI must be on `PATH`
(pyrubberband calls it), and ffmpeg for MP3 sources.

## The constitution (machine-checked)

`lint-imports` must pass; `pytest` must pass. The contract lives in `pyproject.toml`:
- **The renderer takes grid results (tempo, first downbeat, bar frames) and never computes analysis.**
  No librosa, no dj-score. Its one detector, `loader.detect_first_downbeat`, is the fallback when a spec has
  no `first_beat_sample`; the plan is for the grid to come from dj-score's analysis in the music library.
- The spec JSON is the interface. A spec finds its tracks and renders from its own location (`paths.py`).

## How changes are verified

**A refactor must not change the audio.** Before changing anything, render a few transitions; after, render
them again and compare byte for byte (renders are deterministic, including Rubber Band). A change that *is* meant
to change the audio (e.g. a different grid) is A/B'd by ear and, once it exists, by an objective verify step.

**This repo is code only.** Tests use synthetic tones. Tracks, specs and renders live in the shared music library
(sibling `../music`, local only, never pushed: commercial audio).
