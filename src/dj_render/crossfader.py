"""Crossfade engine — build transitions from scratch using numpy math."""

import numpy as np


# --- Fade curve generators ---

def linear_fade(n_samples, **kwargs):
    """Linear ramp from 1.0 to 0.0.

    Returns:
        tuple: (fade_out, fade_in) curves of length n_samples
    """
    fade_out = np.linspace(1.0, 0.0, n_samples)
    fade_in = np.linspace(0.0, 1.0, n_samples)
    return fade_out, fade_in


def equal_power_fade(n_samples, **kwargs):
    """Equal-power (constant power) crossfade — no volume dip in the middle.

    Uses cos/sin curves so that fade_out^2 + fade_in^2 = 1 at every sample.

    Returns:
        tuple: (fade_out, fade_in) curves of length n_samples
    """
    t = np.linspace(0, np.pi / 2, n_samples)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    return fade_out, fade_in


def s_curve_fade(n_samples, **kwargs):
    """S-curve (smooth) crossfade — slow start, fast middle, slow end.

    Uses a raised cosine for a natural-feeling transition.

    Returns:
        tuple: (fade_out, fade_in) curves of length n_samples
    """
    t = np.linspace(0, np.pi, n_samples)
    fade_out = (1 + np.cos(t)) / 2
    fade_in = (1 - np.cos(t)) / 2
    return fade_out, fade_in


def exp_fade(n_samples, power=3.0):
    """Exponential/polynomial crossfade — incoming ramps up fast, outgoing lingers.

    Both tracks stay at higher combined volume during the overlap,
    filling gaps when both sources are quiet. Power controls the curve
    shape: higher = faster ramp-in, slower fade-out.

    At the midpoint with power=3: fade_out ≈ 0.87, fade_in ≈ 0.87
    (both near full volume, gentle overlap).

    Returns:
        tuple: (fade_out, fade_in) curves of length n_samples
    """
    t = np.linspace(0, 1, n_samples)
    fade_in = t ** (1.0 / power)   # concave — ramps up fast
    fade_out = (1 - t) ** (1.0 / power)  # concave — lingers long
    return fade_out.astype(np.float32), fade_in.astype(np.float32)


FADE_CURVES = {
    "linear": linear_fade,
    "equal_power": equal_power_fade,
    "s_curve": s_curve_fade,
    "exp": exp_fade,
}


def get_fade_curves(fade_type, n_samples, **kwargs):
    """Get fade-out and fade-in curves by name.

    Args:
        fade_type: one of "linear", "equal_power", "s_curve", "exp"
        n_samples: length of the fade in samples
        **kwargs: passed to the fade function (e.g. power for exp)

    Returns:
        tuple: (fade_out, fade_in) arrays
    """
    if fade_type not in FADE_CURVES:
        raise ValueError(
            f"Unknown fade type '{fade_type}'. "
            f"Available: {list(FADE_CURVES.keys())}"
        )
    return FADE_CURVES[fade_type](n_samples, **kwargs)


# --- Crossfade builders ---

def crossfade(track_out, track_in, overlap_samples, fade_type="equal_power", **fade_kwargs):
    """Build a crossfade between two tracks.

    track_out's last `overlap_samples` are faded out while
    track_in's first `overlap_samples` are faded in, then summed.

    Args:
        track_out: outgoing track audio array (full cued region)
        track_in: incoming track audio array (full cued region)
        overlap_samples: number of samples in the overlap zone
        fade_type: "linear", "equal_power", "s_curve", or "exp"
        **fade_kwargs: passed to fade function (e.g. power for exp)

    Returns:
        numpy array: the complete joined audio
            [track_out solo] + [crossfaded overlap] + [track_in solo]
    """
    if overlap_samples <= 0:
        return np.concatenate([track_out, track_in])

    # Clamp overlap to available audio
    overlap_samples = min(overlap_samples, len(track_out), len(track_in))

    # Split into solo and overlap regions
    out_solo = track_out[:-overlap_samples]
    out_overlap = track_out[-overlap_samples:]
    in_overlap = track_in[:overlap_samples]
    in_solo = track_in[overlap_samples:]

    # Apply fade curves
    fade_out, fade_in = get_fade_curves(fade_type, overlap_samples, **fade_kwargs)
    mixed_overlap = (out_overlap * fade_out) + (in_overlap * fade_in)

    return np.concatenate([out_solo, mixed_overlap, in_solo])


def clean_cut(track_out, track_in):
    """Hard cut — no overlap, no fade. One track ends, the next begins.

    Args:
        track_out: outgoing track audio (trimmed to cue point)
        track_in: incoming track audio (trimmed to cue point)

    Returns:
        concatenated audio
    """
    return np.concatenate([track_out, track_in])


def bars_to_samples(n_bars, bpm, sr):
    """Convert a number of bars to samples.

    Assumes 4/4 time (4 beats per bar).

    Args:
        n_bars: number of bars
        bpm: tempo in BPM
        sr: sample rate

    Returns:
        number of samples (int)
    """
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4
    return int(n_bars * seconds_per_bar * sr)
