"""Mix engine — reads a mix spec and produces a continuous audio file."""

import json
import os

import numpy as np
import soundfile as sf

from .loader import load_and_prepare, trim_to_bars, apply_loop
from .stretcher import time_stretch, recompute_bars_after_stretch
from .crossfader import crossfade, clean_cut, bars_to_samples, get_fade_curves
from .effects import apply_gain, apply_eq, apply_3band_ramp, split_3band
from .paths import default_render_path, resolve_tracks_dir


def load_spec(spec_path):
    """Load a mix spec JSON file, resolving its `tracks_dir` from the spec's location
    (`paths.resolve_tracks_dir`) and remembering where the spec lives (`_spec_path`).

    Returns:
        dict: the parsed mix specification
    """
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    spec["_spec_path"] = os.path.abspath(spec_path)
    spec["tracks_dir"] = resolve_tracks_dir(spec_path, spec.get("tracks_dir"),
                                            [t["file"] for t in spec.get("tracks", [])])
    return spec


def prepare_track(track_spec, tracks_dir, target_sr):
    """Load, analyze, stretch, and trim a single track.

    Returns:
        dict: prepared track data with keys: audio, sr, bpm, bar_frames, spec
    """
    print(f"  Loading: {track_spec['name']}...")
    data = load_and_prepare(track_spec, tracks_dir, target_sr)
    first_beat_sec = data.get("first_beat", 0) / target_sr
    print(f"  First downbeat: {first_beat_sec:.3f}s | Bars: {len(data['bar_frames'])}")

    # Time-stretch to target BPM
    original_bpm = track_spec["original_bpm"]
    target_bpm = track_spec["target_bpm"]
    if abs(original_bpm - target_bpm) >= 0.01:
        print(f"  Stretching: {original_bpm:.2f} -> {target_bpm:.2f} BPM...")
        data["audio"] = time_stretch(
            data["audio"], data["sr"], original_bpm, target_bpm
        )
        data["bar_frames"] = recompute_bars_after_stretch(
            data["bar_frames"], original_bpm, target_bpm
        )
        data["bpm"] = target_bpm

    # Trim to cue points
    cue_in = track_spec.get("cue_in_bar", 0)
    cue_out = track_spec.get("cue_out_bar")
    if cue_out is None:
        cue_out = len(data["bar_frames"])
    print(f"  Trimming: bars {cue_in} to {cue_out}...")
    data["audio"] = trim_to_bars(data["audio"], data["bar_frames"], cue_in, cue_out)

    # Rebase bar_frames to the trimmed audio (use integer bar indices for slicing)
    cue_in_int = int(cue_in)
    cue_out_int = int(cue_out) + (1 if cue_out != int(cue_out) else 0)
    if cue_in_int < len(data["bar_frames"]):
        offset = data["bar_frames"][cue_in_int]
        end_idx = min(cue_out_int, len(data["bar_frames"]))
        data["bar_frames"] = data["bar_frames"][cue_in_int:end_idx] - offset
    else:
        data["bar_frames"] = np.array([0])

    # Apply loops if specified
    for loop_pos in ("loop_start", "loop_end"):
        loop_spec = track_spec.get(loop_pos)
        if loop_spec:
            position = "start" if loop_pos == "loop_start" else "end"
            data["audio"], data["bar_frames"] = apply_loop(
                data["audio"], data["bar_frames"], loop_spec, position, target_sr)

    # Apply gain
    gain_db = track_spec.get("gain_db", 0)
    if gain_db != 0:
        print(f"  Gain: {gain_db:+.1f} dB")
        data["audio"] = apply_gain(data["audio"], gain_db)

    return data


def _build_ramp_steps(start_bpm, end_bpm, overlap_bars, step_bars=2):
    """Build BPM values for each chunk of a tempo ramp.

    Returns list of (chunk_bars, target_bpm) tuples.
    """
    n_steps = overlap_bars // step_bars
    remainder = overlap_bars % step_bars

    if n_steps <= 1:
        return [(overlap_bars, end_bpm)]

    # How many steps does the actual ramp take?
    bpm_diff = end_bpm - start_bpm
    bpm_per_step = bpm_diff / (n_steps - 1) if n_steps > 1 else 0

    steps = []
    for i in range(n_steps):
        bpm = start_bpm + i * bpm_per_step
        # Clamp final step to exact end_bpm
        if i == n_steps - 1:
            bpm = end_bpm
        bars = step_bars + (remainder if i == n_steps - 1 else 0)
        steps.append((bars, round(bpm, 4)))

    return steps


def _stretch_overlap_with_ramp(audio, native_bpm, ramp_steps, sr):
    """Stretch an overlap region chunk-by-chunk following a BPM ramp.

    Args:
        audio: the overlap audio at native_bpm
        native_bpm: the track's original BPM
        ramp_steps: list of (chunk_bars, target_bpm) from _build_ramp_steps
        sr: sample rate

    Returns:
        stretched audio array
    """
    samples_per_bar = (60.0 / native_bpm) * 4 * sr
    pos = 0
    chunks = []

    for chunk_bars, target_bpm in ramp_steps:
        chunk_samples = int(chunk_bars * samples_per_bar)
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]

        if abs(native_bpm - target_bpm) >= 0.01 and len(chunk) > 0:
            chunk = time_stretch(chunk, sr, native_bpm, target_bpm)

        chunks.append(chunk)
        pos = end

    # Include any remaining audio (rounding leftovers)
    if pos < len(audio):
        chunks.append(audio[pos:])

    return np.concatenate(chunks) if chunks else audio


def _apply_3band(audio, sr, spec_key, transition_spec):
    """Apply 3-band EQ from transition spec if present.

    Spec format: {"low": [start, end], "mid": [start, end], "high": [start, end]}
    Gains are linear: 0.0=kill, 1.0=unity.

    Returns:
        processed audio, or original if no 3-band spec
    """
    band_spec = transition_spec.get(spec_key)
    if not band_spec:
        return audio

    bands_start = {k: v[0] if isinstance(v, list) else v for k, v in band_spec.items()
                   if k in ('low', 'mid', 'high')}
    bands_end = {k: v[1] if isinstance(v, list) else v for k, v in band_spec.items()
                 if k in ('low', 'mid', 'high')}

    low_freq = band_spec.get("low_freq", 300)
    high_freq = band_spec.get("high_freq", 3000)

    is_ramped = any(isinstance(v, list) and len(v) == 2 and v[0] != v[1]
                    for k, v in band_spec.items() if k in ('low', 'mid', 'high'))

    if is_ramped:
        power = band_spec.get("power", 1.0)
        hold = band_spec.get("hold", 0.0)
        hold_bars = band_spec.get("hold_bars")
        if hold_bars is not None:
            overlap_bars = transition_spec.get("overlap_bars", 1)
            hold = hold_bars / overlap_bars
        result = apply_3band_ramp(audio, sr, bands_start, bands_end, low_freq, high_freq, power, hold)
        extras = []
        if power != 1.0:
            extras.append(f"power={power}")
        if hold > 0:
            extras.append(f"hold={hold:.0%}")
            if hold_bars is not None:
                extras.append(f"({hold_bars} bars)")
        extra_str = f" {' '.join(extras)}" if extras else ""
        print(f"    3-band {spec_key}: low {bands_start.get('low',1):.1f}->{bands_end.get('low',1):.1f} "
              f"mid {bands_start.get('mid',1):.1f}->{bands_end.get('mid',1):.1f} "
              f"high {bands_start.get('high',1):.1f}->{bands_end.get('high',1):.1f}{extra_str}")
    else:
        from .effects import apply_3band_eq
        result = apply_3band_eq(audio, sr,
                                bands_start.get('low', 1.0),
                                bands_start.get('mid', 1.0),
                                bands_start.get('high', 1.0),
                                low_freq, high_freq)
        print(f"    3-band {spec_key}: low={bands_start.get('low',1):.1f} "
              f"mid={bands_start.get('mid',1):.1f} high={bands_start.get('high',1):.1f}")

    return result


def _build_lane_envelope(automation, lane_name, overlap_bars, overlap_samples, default_value=1.0):
    """Build a sample-level gain envelope from automation breakpoints.

    Args:
        automation: automation dict from transition spec
        lane_name: key to look up (e.g. "outgoing_low")
        overlap_bars: total overlap in bars (for x-axis scaling)
        overlap_samples: total overlap in samples (output length)
        default_value: constant value if lane is absent

    Returns:
        numpy float32 array of length overlap_samples
    """
    points = automation.get(lane_name)
    if points is None:
        return np.full(overlap_samples, default_value, dtype=np.float32)

    bars = np.array([p[0] for p in points])
    gains = np.array([p[1] for p in points])
    samples_per_bar = overlap_samples / overlap_bars
    sample_positions = np.arange(overlap_samples) / samples_per_bar
    return np.interp(sample_positions, bars, gains).astype(np.float32)


def execute_transition(track_out_data, track_in_data, transition_spec, sr):
    """Execute a single transition between two prepared tracks.

    Supports optional tempo_ramp for gradual BPM matching during crossfade.

    Args:
        track_out_data: prepared outgoing track dict
        track_in_data: prepared incoming track dict
        transition_spec: transition parameters from mix spec
        sr: sample rate

    Returns:
        numpy array: the joined audio
    """
    trans_type = transition_spec.get("type", "crossfade")
    out_name = track_out_data["spec"]["name"]
    in_name = track_in_data["spec"]["name"]

    if trans_type == "clean_cut":
        print(f"  Transition: {out_name} -> {in_name} [clean cut]")
        return clean_cut(track_out_data["audio"], track_in_data["audio"])

    # Crossfade
    overlap_bars = transition_spec.get("overlap_bars", 16)
    fade_type = transition_spec.get("fade_type", "equal_power")
    tempo_ramp = transition_spec.get("tempo_ramp")

    out_audio = track_out_data["audio"]
    in_audio = track_in_data["audio"]
    out_bpm = track_out_data["bpm"]
    in_bpm = track_in_data["bpm"]

    if tempo_ramp:
        if transition_spec.get("fade_offset_bars", 0) > 0 or \
           transition_spec.get("fade_bars") is not None:
            print("    WARNING: fade_offset_bars/fade_bars ignored with tempo_ramp (use automation instead)")
        # Tempo ramp transition: stretch chunks to match BPMs
        step_bars = tempo_ramp.get("step_bars", 2)
        start_bpm = tempo_ramp.get("start_bpm", out_bpm)
        end_bpm = tempo_ramp.get("end_bpm", in_bpm)
        ramp_bars = tempo_ramp.get("ramp_bars", overlap_bars)

        # Build full ramp steps across ramp_bars
        full_ramp_steps = _build_ramp_steps(start_bpm, end_bpm, ramp_bars, step_bars)
        step_bpms = [f"{bpm:.1f}" for _, bpm in full_ramp_steps]

        pre_ramp_bars = ramp_bars - overlap_bars
        if pre_ramp_bars < 0:
            pre_ramp_bars = 0

        print(
            f"  Transition: {out_name} -> {in_name} "
            f"[{fade_type} crossfade, {overlap_bars} bars overlap, "
            f"tempo ramp {start_bpm:.1f}->{end_bpm:.1f} over {ramp_bars} bars "
            f"in {step_bars}-bar steps]"
        )
        print(f"    Ramp: {' -> '.join(step_bpms)}")
        if pre_ramp_bars > 0:
            print(f"    Pre-ramp: {pre_ramp_bars} bars solo, then {overlap_bars} bars crossfade")

        if pre_ramp_bars > 0:
            # Split ramp steps into pre-overlap (solo) and overlap portions
            pre_steps = []
            overlap_steps = []
            bars_counted = 0
            for chunk_bars, bpm in full_ramp_steps:
                remaining_pre = pre_ramp_bars - bars_counted
                if remaining_pre <= 0:
                    overlap_steps.append((chunk_bars, bpm))
                elif chunk_bars <= remaining_pre:
                    pre_steps.append((chunk_bars, bpm))
                else:
                    pre_steps.append((remaining_pre, bpm))
                    overlap_steps.append((chunk_bars - remaining_pre, bpm))
                bars_counted += chunk_bars

            # Extract outgoing audio: solo | pre-ramp | overlap
            pre_ramp_samples = bars_to_samples(pre_ramp_bars, out_bpm, sr)
            out_overlap_samples = bars_to_samples(overlap_bars, out_bpm, sr)
            total_out = min(pre_ramp_samples + out_overlap_samples, len(out_audio))
            out_overlap_samples = min(out_overlap_samples, len(out_audio))

            out_solo = out_audio[:-total_out] if total_out < len(out_audio) else np.array([], dtype=np.float32)
            out_pre_ramp = out_audio[-total_out:-out_overlap_samples]
            out_overlap = out_audio[-out_overlap_samples:]

            # Stretch pre-ramp section (outgoing solo, tempo-ramped)
            out_pre_stretched = _stretch_overlap_with_ramp(
                out_pre_ramp, out_bpm, pre_steps, sr)

            # Stretch overlap portions for both tracks
            out_overlap_stretched = _stretch_overlap_with_ramp(
                out_overlap, out_bpm, overlap_steps, sr)

            in_overlap_samples = bars_to_samples(overlap_bars, in_bpm, sr)
            in_overlap_samples = min(in_overlap_samples, len(in_audio))
            in_overlap = in_audio[:in_overlap_samples]
            in_solo = in_audio[in_overlap_samples:]

            in_overlap_stretched = _stretch_overlap_with_ramp(
                in_overlap, in_bpm, overlap_steps, sr)

            # Match lengths for crossfade
            min_len = min(len(out_overlap_stretched), len(in_overlap_stretched))
            out_overlap_stretched = out_overlap_stretched[:min_len]
            in_overlap_stretched = in_overlap_stretched[:min_len]

            # Apply EQ to stretched overlap regions
            eq_out = transition_spec.get("eq_outgoing")
            if eq_out:
                out_overlap_stretched = apply_eq(out_overlap_stretched, sr, eq_out)

            eq_in = transition_spec.get("eq_incoming")
            if eq_in:
                in_original = in_overlap_stretched.copy()
                in_eq = apply_eq(in_overlap_stretched, sr, eq_in)
                if transition_spec.get("eq_incoming_ramp", False):
                    t = np.linspace(0.0, 1.0, min_len).astype(np.float32)
                    ramp_power = transition_spec.get("eq_ramp_power", 1.0)
                    ramp = (1.0 - t) ** ramp_power
                    in_overlap_stretched = in_eq * ramp + in_original * (1.0 - ramp)
                    curve_name = "exp" if ramp_power > 1 else "log" if ramp_power < 1 else "linear"
                    print(f"    EQ incoming: {curve_name} ramp (power={ramp_power}) over {min_len/sr:.1f}s")
                else:
                    in_overlap_stretched = in_eq

            # Apply 3-band EQ if specified
            out_overlap_stretched = _apply_3band(out_overlap_stretched, sr, "eq_3band_outgoing", transition_spec)
            in_overlap_stretched = _apply_3band(in_overlap_stretched, sr, "eq_3band_incoming", transition_spec)

            # Apply fade curves
            fade_kwargs = {}
            fade_power = transition_spec.get("fade_power")
            if fade_power is not None:
                fade_kwargs["power"] = fade_power
                print(f"    Fade power: {fade_power}")

            if transition_spec.get("outgoing_hold", False):
                fade_out = np.ones(min_len, dtype=np.float32)
                _, fade_in = get_fade_curves(fade_type, min_len, **fade_kwargs)
                mixed_overlap = (out_overlap_stretched * fade_out) + (in_overlap_stretched * fade_in)
                print(f"    Outgoing hold: full volume, slam cut at end")
            else:
                fade_out, fade_in = get_fade_curves(fade_type, min_len, **fade_kwargs)
                mixed_overlap = (out_overlap_stretched * fade_out) + (in_overlap_stretched * fade_in)

            pre_dur = len(out_pre_stretched) / sr
            xfade_dur = min_len / sr
            print(f"    Pre-ramp: {pre_dur:.1f}s solo | Crossfade: {xfade_dur:.1f}s")

            return np.concatenate([out_solo, out_pre_stretched, mixed_overlap, in_solo])

        # Standard tempo ramp (ramp_bars == overlap_bars)
        ramp_steps = full_ramp_steps

        # Extract overlap regions at each track's native BPM
        out_overlap_samples = bars_to_samples(overlap_bars, out_bpm, sr)
        in_overlap_samples = bars_to_samples(overlap_bars, in_bpm, sr)

        out_overlap_samples = min(out_overlap_samples, len(out_audio))
        in_overlap_samples = min(in_overlap_samples, len(in_audio))

        out_solo = out_audio[:-out_overlap_samples]
        out_overlap = out_audio[-out_overlap_samples:]
        in_overlap = in_audio[:in_overlap_samples]
        in_solo = in_audio[in_overlap_samples:]

        # Stretch each overlap chunk to the ramp BPM
        out_stretched = _stretch_overlap_with_ramp(
            out_overlap, out_bpm, ramp_steps, sr)
        in_stretched = _stretch_overlap_with_ramp(
            in_overlap, in_bpm, ramp_steps, sr)

        # Match lengths for crossfade
        min_len = min(len(out_stretched), len(in_stretched))
        out_stretched = out_stretched[:min_len]
        in_stretched = in_stretched[:min_len]

        # Apply EQ to stretched overlap regions (post-stretch for correct ramp timing)
        eq_out = transition_spec.get("eq_outgoing")
        if eq_out:
            out_stretched = apply_eq(out_stretched, sr, eq_out)

        eq_in = transition_spec.get("eq_incoming")
        if eq_in:
            in_original = in_stretched.copy()
            in_stretched_eq = apply_eq(in_stretched, sr, eq_in)
            if transition_spec.get("eq_incoming_ramp", False):
                t = np.linspace(0.0, 1.0, min_len).astype(np.float32)
                ramp_power = transition_spec.get("eq_ramp_power", 1.0)
                ramp = (1.0 - t) ** ramp_power
                in_stretched = in_stretched_eq * ramp + in_original * (1.0 - ramp)
                curve_name = "exp" if ramp_power > 1 else "log" if ramp_power < 1 else "linear"
                print(f"    EQ incoming: {curve_name} ramp (power={ramp_power}) over {min_len/sr:.1f}s")
            else:
                in_stretched = in_stretched_eq

        # Apply 3-band EQ if specified
        out_stretched = _apply_3band(out_stretched, sr, "eq_3band_outgoing", transition_spec)
        in_stretched = _apply_3band(in_stretched, sr, "eq_3band_incoming", transition_spec)

        # Apply fade curves
        fade_kwargs = {}
        fade_power = transition_spec.get("fade_power")
        if fade_power is not None:
            fade_kwargs["power"] = fade_power
            print(f"    Fade power: {fade_power}")

        if transition_spec.get("outgoing_hold", False):
            fade_out = np.ones(min_len, dtype=np.float32)
            _, fade_in = get_fade_curves(fade_type, min_len, **fade_kwargs)
            mixed_overlap = (out_stretched * fade_out) + (in_stretched * fade_in)
            print(f"    Outgoing hold: vocals at full volume, slam cut at end")
        else:
            fade_out, fade_in = get_fade_curves(fade_type, min_len, **fade_kwargs)
            mixed_overlap = (out_stretched * fade_out) + (in_stretched * fade_in)

        ramp_dur = min_len / sr
        print(f"    Overlap duration: {ramp_dur:.1f}s")

        return np.concatenate([out_solo, mixed_overlap, in_solo])

    # Standard crossfade (no tempo ramp)
    bpm = in_bpm
    overlap_samples = bars_to_samples(overlap_bars, bpm, sr)
    print(
        f"  Transition: {out_name} -> {in_name} "
        f"[{fade_type} crossfade, {overlap_bars} bars, "
        f"{overlap_samples / sr:.1f}s]"
    )

    # Per-band EQ automation: check if automation has EQ lane keys
    EQ_LANE_KEYS = {"outgoing_low", "outgoing_mid", "outgoing_high",
                    "incoming_low", "incoming_mid", "incoming_high"}
    automation = transition_spec.get("automation")
    if automation and EQ_LANE_KEYS & set(automation.keys()):
        # Warn about incompatible features being overridden
        if transition_spec.get("eq_3band_outgoing"):
            print("    WARNING: eq_3band_outgoing ignored (per-band automation lanes active)")
        if transition_spec.get("eq_3band_incoming"):
            print("    WARNING: eq_3band_incoming ignored (per-band automation lanes active)")
        if transition_spec.get("eq_outgoing"):
            print("    WARNING: eq_outgoing ignored (per-band automation lanes active)")
        if transition_spec.get("eq_incoming"):
            print("    WARNING: eq_incoming ignored (per-band automation lanes active)")
        if transition_spec.get("outgoing_hold"):
            print("    WARNING: outgoing_hold ignored (per-band automation lanes active)")
        if transition_spec.get("fade_type") and transition_spec.get("fade_type") != "equal_power":
            print("    WARNING: fade_type ignored (fader lanes control volume)")
        if transition_spec.get("fade_power") is not None:
            print("    WARNING: fade_power ignored (fader lanes control volume)")
        if transition_spec.get("fade_offset_bars", 0) > 0:
            print("    WARNING: fade_offset_bars ignored (fader lanes control volume)")
        if transition_spec.get("overlap_gain_hump_db"):
            print("    WARNING: overlap_gain_hump_db ignored (automation lanes control gain)")
        if transition_spec.get("tempo_ramp"):
            print("    WARNING: tempo_ramp not supported with per-band automation; using standard crossfade path")

        # Split overlap regions into 3 bands
        low_freq = automation.get("low_freq", 300)
        high_freq = automation.get("high_freq", 3000)

        out_overlap = out_audio[-overlap_samples:]
        in_overlap = in_audio[:overlap_samples]
        out_solo = out_audio[:-overlap_samples]
        in_solo = in_audio[overlap_samples:]

        out_low, out_mid, out_high = split_3band(out_overlap, sr, low_freq, high_freq)
        in_low, in_mid, in_high = split_3band(in_overlap, sr, low_freq, high_freq)

        # Build per-band gain envelopes
        out_low_env = _build_lane_envelope(automation, "outgoing_low", overlap_bars, overlap_samples)
        out_mid_env = _build_lane_envelope(automation, "outgoing_mid", overlap_bars, overlap_samples)
        out_high_env = _build_lane_envelope(automation, "outgoing_high", overlap_bars, overlap_samples)
        in_low_env = _build_lane_envelope(automation, "incoming_low", overlap_bars, overlap_samples)
        in_mid_env = _build_lane_envelope(automation, "incoming_mid", overlap_bars, overlap_samples)
        in_high_env = _build_lane_envelope(automation, "incoming_high", overlap_bars, overlap_samples)

        # Apply per-band gains and reassemble
        out_eq = (out_low * out_low_env) + (out_mid * out_mid_env) + (out_high * out_high_env)
        in_eq = (in_low * in_low_env) + (in_mid * in_mid_env) + (in_high * in_high_env)

        # Build fader envelopes (outgoing_fader > outgoing > default 1.0)
        out_fader_key = "outgoing_fader" if "outgoing_fader" in automation else "outgoing"
        in_fader_key = "incoming_fader" if "incoming_fader" in automation else "incoming"
        out_fader = _build_lane_envelope(automation, out_fader_key, overlap_bars, overlap_samples)
        in_fader = _build_lane_envelope(automation, in_fader_key, overlap_bars, overlap_samples)

        # Mix: (EQ'd outgoing * fader) + (EQ'd incoming * fader)
        mixed_overlap = (out_eq * out_fader + in_eq * in_fader).astype(np.float32)

        # Print lane summary
        active_lanes = [k for k in EQ_LANE_KEYS if automation.get(k)]
        fader_lanes = [k for k in ("outgoing_fader", "incoming_fader", "outgoing", "incoming")
                       if automation.get(k)]
        lane_details = []
        for lane in sorted(active_lanes):
            pts = automation[lane]
            lane_details.append(f"{lane}({len(pts)}pts)")
        for lane in fader_lanes:
            pts = automation[lane]
            lane_details.append(f"{lane}({len(pts)}pts)")
        print(f"    Per-band automation: {', '.join(lane_details)}")
        print(f"    Crossover: {low_freq}/{high_freq} Hz")

        return np.concatenate([out_solo, mixed_overlap, in_solo])

    # Apply transition EQ to the overlap regions if specified
    eq_out = transition_spec.get("eq_outgoing")
    if eq_out:
        tail_start = max(0, len(out_audio) - overlap_samples)
        tail = out_audio[tail_start:]
        tail = apply_eq(tail, sr, eq_out)
        out_audio = np.copy(out_audio)
        out_audio[tail_start:] = tail

    eq_in = transition_spec.get("eq_incoming")
    if eq_in:
        head_original = in_audio[:overlap_samples].copy()
        head_eq = apply_eq(head_original, sr, eq_in)
        if transition_spec.get("eq_incoming_ramp", False):
            # Ramp from fully EQ'd (start) to unfiltered (end of overlap)
            t = np.linspace(0.0, 1.0, overlap_samples).astype(np.float32)
            ramp_power = transition_spec.get("eq_ramp_power", 1.0)
            # ramp = EQ strength: 1.0 at start, 0.0 at end
            # power > 1: stays filtered longer (exp), power < 1: opens fast (log)
            ramp = (1.0 - t) ** ramp_power
            head = head_eq * ramp + head_original * (1.0 - ramp)
            curve_name = "exp" if ramp_power > 1 else "log" if ramp_power < 1 else "linear"
            print(f"    EQ incoming: {curve_name} ramp (power={ramp_power}) over {overlap_samples/sr:.1f}s")
        else:
            head = head_eq
        in_audio = np.copy(in_audio)
        in_audio[:overlap_samples] = head

    # Apply 3-band EQ to overlap regions if specified
    eq_3b_out = transition_spec.get("eq_3band_outgoing")
    if eq_3b_out:
        tail_start = max(0, len(out_audio) - overlap_samples)
        tail = _apply_3band(out_audio[tail_start:], sr, "eq_3band_outgoing", transition_spec)
        out_audio = np.copy(out_audio)
        out_audio[tail_start:] = tail

    eq_3b_in = transition_spec.get("eq_3band_incoming")
    if eq_3b_in:
        head = _apply_3band(in_audio[:overlap_samples], sr, "eq_3band_incoming", transition_spec)
        in_audio = np.copy(in_audio)
        in_audio[:overlap_samples] = head

    fade_kwargs = {}
    fade_power = transition_spec.get("fade_power")
    if fade_power is not None:
        fade_kwargs["power"] = fade_power
        print(f"    Fade power: {fade_power}")

    # Decoupled EQ/Fader: EQ spans full overlap, fader moves in a sub-region
    fade_offset_bars = transition_spec.get("fade_offset_bars", 0)
    fade_bars_count = transition_spec.get("fade_bars", overlap_bars)

    # Clamp to not exceed overlap
    if fade_offset_bars + fade_bars_count > overlap_bars:
        fade_bars_count = overlap_bars - fade_offset_bars
        print(f"    WARNING: fade_offset_bars + fade_bars clamped to overlap_bars ({overlap_bars})")

    if fade_offset_bars > 0 or fade_bars_count < overlap_bars:
        # Decoupled fader path — three zones
        if transition_spec.get("outgoing_hold", False):
            print("    WARNING: outgoing_hold ignored with decoupled fader (fade_offset_bars/fade_bars)")

        if not transition_spec.get("eq_3band_outgoing") and not transition_spec.get("eq_3band_incoming") \
           and not transition_spec.get("eq_outgoing") and not transition_spec.get("eq_incoming"):
            print("    WARNING: decoupled fader with no EQ — pre-fade zone will be loud (+6 dB)")

        pre_samples = bars_to_samples(fade_offset_bars, bpm, sr)
        fade_samples = bars_to_samples(fade_bars_count, bpm, sr)

        out_overlap = out_audio[-overlap_samples:]
        in_overlap = in_audio[:overlap_samples]
        out_solo = out_audio[:-overlap_samples]
        in_solo = in_audio[overlap_samples:]

        # Zone 1: Pre-fade — both at full volume, sum
        pre_mix = out_overlap[:pre_samples] + in_overlap[:pre_samples]

        # Zone 2: Fade — outgoing fades out, incoming stays at 1.0
        fade_end = pre_samples + fade_samples
        fade_out, _ = get_fade_curves(fade_type, fade_samples, **fade_kwargs)
        fade_mix = (out_overlap[pre_samples:fade_end] * fade_out
                    + in_overlap[pre_samples:fade_end])

        # Zone 3: Post-fade — incoming only (if any remainder)
        post_mix = in_overlap[fade_end:]

        mixed_overlap = np.concatenate([pre_mix, fade_mix, post_mix])

        # Gain hump scoped to fade zone only
        hump_db = transition_spec.get("overlap_gain_hump_db")
        if hump_db and fade_samples > 0:
            t = np.linspace(0, np.pi, fade_samples)
            hump = np.sin(t)
            gain_curve = (10.0 ** (hump_db * hump / 20.0)).astype(np.float32)
            mixed_overlap[pre_samples:fade_end] *= gain_curve
            print(f"    Overlap gain hump: +{hump_db:.1f} dB peak (fade zone only)")

        pre_dur = pre_samples / sr
        fade_dur = fade_samples / sr
        post_dur = len(post_mix) / sr
        print(f"    Decoupled fader: {fade_offset_bars} bars pre-fade ({pre_dur:.1f}s) | "
              f"{fade_bars_count} bars fade ({fade_dur:.1f}s) | "
              f"{overlap_bars - fade_offset_bars - fade_bars_count} bars post-fade ({post_dur:.1f}s)")

        return np.concatenate([out_solo, mixed_overlap, in_solo])

    if transition_spec.get("outgoing_hold", False):
        # Hold outgoing at full volume, only fade in incoming, then hard-cut
        out_solo = out_audio[:-overlap_samples]
        out_overlap = out_audio[-overlap_samples:]
        in_overlap = in_audio[:overlap_samples]
        in_solo = in_audio[overlap_samples:]

        _, fade_in = get_fade_curves(fade_type, overlap_samples, **fade_kwargs)
        mixed_overlap = out_overlap + (in_overlap * fade_in)
        print(f"    Outgoing hold: vocals at full volume, slam cut at end")

        return np.concatenate([out_solo, mixed_overlap, in_solo])

    # Fader automation: breakpoint-based volume envelopes override fade curves
    automation = transition_spec.get("automation")
    if automation:
        out_solo = out_audio[:-overlap_samples]
        out_overlap = out_audio[-overlap_samples:]
        in_overlap = in_audio[:overlap_samples]
        in_solo = in_audio[overlap_samples:]

        samples_per_bar = overlap_samples / overlap_bars

        # Build gain envelopes from breakpoints via linear interpolation
        out_points = automation.get("outgoing", [[0, 1.0], [overlap_bars, 0.0]])
        in_points = automation.get("incoming", [[0, 0.0], [overlap_bars, 1.0]])

        out_bars = np.array([p[0] for p in out_points])
        out_gains = np.array([p[1] for p in out_points])
        in_bars = np.array([p[0] for p in in_points])
        in_gains = np.array([p[1] for p in in_points])

        # Interpolate to sample-level envelopes
        sample_positions = np.arange(overlap_samples) / samples_per_bar
        out_env = np.interp(sample_positions, out_bars, out_gains).astype(np.float32)
        in_env = np.interp(sample_positions, in_bars, in_gains).astype(np.float32)

        mixed_overlap = (out_overlap * out_env) + (in_overlap * in_env)

        n_out = len(out_points)
        n_in = len(in_points)
        print(f"    Fader automation: {n_out} outgoing breakpoints, {n_in} incoming breakpoints")

        return np.concatenate([out_solo, mixed_overlap, in_solo])

    result = crossfade(out_audio, in_audio, overlap_samples, fade_type, **fade_kwargs)

    # Overlap gain hump: sin-curve boost that peaks at crossfade midpoint
    hump_db = transition_spec.get("overlap_gain_hump_db")
    if hump_db:
        out_solo_len = len(out_audio) - overlap_samples
        t = np.linspace(0, np.pi, overlap_samples)
        hump = np.sin(t)
        gain_curve = (10.0 ** (hump_db * hump / 20.0)).astype(np.float32)
        result[out_solo_len:out_solo_len + overlap_samples] *= gain_curve
        print(f"    Overlap gain hump: +{hump_db:.1f} dB peak at midpoint")

    return result


def render_mix(spec, tracks_dir=None, output_path=None):
    """Render a complete mix from a spec.

    Args:
        spec: parsed mix specification dict
        tracks_dir: override for tracks directory (default: from spec)
        output_path: override for output file path (default: from spec)

    Returns:
        numpy array: the rendered mix audio
    """
    if tracks_dir is None:
        tracks_dir = spec.get("tracks_dir", ".")
    if output_path is None:
        output_path = default_render_path(spec.get("_spec_path"),
                                          os.path.basename(spec.get("output_file", "mix.wav")))

    sr = spec.get("sample_rate", 44100)
    tracks = spec["tracks"]
    transitions = spec.get("transitions", [])

    # Build a lookup from track id to transition spec
    trans_lookup = {}
    for t in transitions:
        key = (t["from_track"], t["to_track"])
        trans_lookup[key] = t

    # Prepare all tracks
    print(f"Preparing {len(tracks)} tracks...")
    prepared = []
    for track_spec in tracks:
        data = prepare_track(track_spec, tracks_dir, sr)
        prepared.append(data)
        print(f"  -> {track_spec['name']}: {len(data['audio']) / sr:.1f}s")

    # Execute transitions sequentially
    print(f"\nExecuting {len(transitions)} transitions...")
    result = prepared[0]["audio"]

    for i in range(1, len(prepared)):
        prev_id = tracks[i - 1]["id"]
        curr_id = tracks[i]["id"]
        trans_spec = trans_lookup.get(
            (prev_id, curr_id),
            {"type": "clean_cut"},  # default to clean cut if no transition defined
        )

        # Build temporary track dicts with the current state of audio
        out_data = {**prepared[i - 1], "audio": result}
        in_data = prepared[i]

        # For the transition, we need the outgoing track to be just result,
        # and the incoming track to be the next prepared track
        result = execute_transition(
            {"audio": result, "spec": prepared[i - 1]["spec"], "bpm": prepared[i - 1]["bpm"]},
            in_data,
            trans_spec,
            sr,
        )

    # Normalize to prevent clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        print(f"\nNormalizing (peak was {peak:.3f})...")
        result = result / peak * 0.95

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, result, sr)
    duration = len(result) / sr
    minutes = int(duration // 60)
    seconds = duration % 60
    print(f"\nRendered: {output_path} ({minutes}:{seconds:05.2f})")

    return result


def render_transition(spec, transition_index, tracks_dir=None, output_path=None):
    """Render just a single transition for quick preview/iteration.

    Args:
        spec: parsed mix specification dict
        transition_index: which transition to render (0-indexed)
        tracks_dir: override for tracks directory
        output_path: override for output file path

    Returns:
        numpy array: the transition audio
    """
    if tracks_dir is None:
        tracks_dir = spec.get("tracks_dir", ".")
    if output_path is None:
        output_path = default_render_path(spec.get("_spec_path"), f"transition_{transition_index}.wav")

    sr = spec.get("sample_rate", 44100)
    transition = spec["transitions"][transition_index]

    # Find the two tracks involved
    from_id = transition["from_track"]
    to_id = transition["to_track"]

    from_spec = next(t for t in spec["tracks"] if t["id"] == from_id)
    to_spec = next(t for t in spec["tracks"] if t["id"] == to_id)

    print(f"Rendering transition: {from_spec['name']} -> {to_spec['name']}")

    # Prepare both tracks
    out_data = prepare_track(from_spec, tracks_dir, sr)
    in_data = prepare_track(to_spec, tracks_dir, sr)

    # Execute transition
    result = execute_transition(out_data, in_data, transition, sr)

    # Normalize
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.95

    # Write
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, result, sr)
    duration = len(result) / sr
    print(f"Rendered: {output_path} ({duration:.1f}s)")

    return result
