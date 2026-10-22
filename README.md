# dj-render

Render a DJ mix spec to audio. Time-stretching, beat-aligned crossfades, EQ transitions, gain staging, all
driven by a JSON spec.

dj-render is the rendering half of a pair: [dj-score](https://github.com/chrisaacson69/dj-score) analyses
tracks (beat grid, key, events, spectra), and dj-render executes a mix from grid results. The controller that
plans and drives both is [CyborgDJ](https://github.com/chrisaacson69/cyborgdj). This code was split out of
CyborgDJ's `cyborgdj/engine/` package (CyborgDJ commit `6b105d5`); see that repo's `AUDIT.md` for why.

## Install

```bash
pip install -e .            # numpy, scipy, soundfile, pyrubberband
pip install -e .[dev]       # + pytest, import-linter
```

System tools on `PATH`: the [Rubber Band](https://breakfastquay.com/rubberband/) CLI (time-stretch) and `ffmpeg`
(MP3 decode).

## Use

```python
from dj_render import load_spec, render_mix, render_transition

spec = load_spec("music/CyborgProjects/MyMix/specs/set.json")
render_transition(spec, 0)          # one transition -> <project>/renders/transition_0.wav
render_mix(spec)                    # the whole mix
```

A spec finds its tracks from where the spec file lives: next to it, then the music library root (the nearest
folder with an `analysis/` subfolder), then the project's `tracks/`. Renders go to `<project>/renders/`. The spec
format is documented in CyborgDJ's README.

## Modules

| Module | Role |
|---|---|
| `mix` | orchestrator: reads a spec, prepares tracks, renders transitions and mixes |
| `loader` | audio I/O (soundfile; ffmpeg for MP3), bar grid from tempo + first downbeat, trims and loops |
| `stretcher` | pitch-preserving time-stretch (Rubber Band; ffmpeg `atempo` fallback) |
| `crossfader` | fade curves and overlap mixing |
| `effects` | EQ, filters, gain, 3-band split (scipy) |
| `paths` | where a spec's tracks and renders live |

## Architecture rule

The renderer takes grid results (tempo, first downbeat, bar frames) and never computes analysis: it may not
import librosa or dj-score (`lint-imports`).
