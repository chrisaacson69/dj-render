"""Audio effects for transitions — EQ, gain, filtering via scipy."""

import numpy as np
from scipy.signal import sosfilt, butter, iirpeak


def apply_gain(audio, gain_db):
    """Apply gain in decibels.

    Args:
        audio: audio array
        gain_db: gain adjustment in dB (positive = louder, negative = quieter)

    Returns:
        gained audio array
    """
    factor = 10.0 ** (gain_db / 20.0)
    return audio * factor


def _highpass(freq, sr, q=0.707):
    """Design a highpass filter, return second-order sections."""
    return butter(2, freq, btype='high', fs=sr, output='sos')


def _lowpass(freq, sr, q=0.707):
    """Design a lowpass filter, return second-order sections."""
    return butter(2, freq, btype='low', fs=sr, output='sos')


def _shelf_filter(freq, sr, gain_db, q=0.707, shelf_type='low'):
    """Design a shelving filter using biquad coefficients.

    Args:
        freq: cutoff frequency
        sr: sample rate
        gain_db: gain in dB
        q: Q factor
        shelf_type: 'low' or 'high'

    Returns:
        second-order sections array
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)

    cos_w0 = np.cos(w0)

    if shelf_type == 'low':
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:  # high shelf
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    # Normalize and pack as single second-order section
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    return sos


def _peak_filter(freq, sr, gain_db, q=0.707):
    """Design a peaking EQ filter using biquad coefficients."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)

    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    return sos


def apply_eq(audio, sr, eq_params):
    """Apply EQ using scipy filters.

    Args:
        audio: mono audio array (float32)
        sr: sample rate
        eq_params: list of EQ band dicts, each with:
            - type: "highpass", "lowpass", "low_shelf", "high_shelf", "peak"
            - frequency: center/cutoff frequency in Hz
            - gain_db: gain for shelf/peak filters (ignored for pass filters)
            - q: Q factor (default 0.707)

    Returns:
        EQ'd audio array
    """
    result = audio.copy()

    for band in eq_params:
        band_type = band["type"]
        freq = band["frequency"]
        q = band.get("q", 0.707)
        gain_db = band.get("gain_db", 0)

        if band_type == "highpass":
            sos = _highpass(freq, sr, q)
        elif band_type == "lowpass":
            sos = _lowpass(freq, sr, q)
        elif band_type == "low_shelf":
            sos = _shelf_filter(freq, sr, gain_db, q, 'low')
        elif band_type == "high_shelf":
            sos = _shelf_filter(freq, sr, gain_db, q, 'high')
        elif band_type == "peak":
            sos = _peak_filter(freq, sr, gain_db, q)
        else:
            continue

        result = sosfilt(sos, result).astype(np.float32)

    return result


def split_3band(audio, sr, low_freq=300, high_freq=3000):
    """Split audio into 3 frequency bands: low, mid, high.

    Uses Linkwitz-Riley style crossover (cascaded Butterworth) for
    flat frequency response when bands are summed.

    Args:
        audio: mono audio array
        sr: sample rate
        low_freq: crossover between low and mid (default 300 Hz)
        high_freq: crossover between mid and high (default 3000 Hz)

    Returns:
        tuple: (low, mid, high) audio arrays
    """
    # Low band: LP at low_freq
    lp_low = butter(2, low_freq, btype='low', fs=sr, output='sos')
    low = sosfilt(lp_low, audio).astype(np.float32)

    # High band: HP at high_freq
    hp_high = butter(2, high_freq, btype='high', fs=sr, output='sos')
    high = sosfilt(hp_high, audio).astype(np.float32)

    # Mid band: what's left (subtract low and high from original)
    mid = (audio - low - high).astype(np.float32)

    return low, mid, high


def apply_3band_eq(audio, sr, low_gain=1.0, mid_gain=1.0, high_gain=1.0,
                   low_freq=300, high_freq=3000):
    """Apply 3-band EQ with independent gain per band.

    Gains are linear multipliers: 0.0 = kill, 1.0 = unity, 2.0 = boost 6dB.

    Args:
        audio: mono audio array
        sr: sample rate
        low_gain: gain for low band (0-300 Hz)
        mid_gain: gain for mid band (300-3000 Hz)
        high_gain: gain for high band (3000+ Hz)
        low_freq: low/mid crossover frequency
        high_freq: mid/high crossover frequency

    Returns:
        processed audio array
    """
    low, mid, high = split_3band(audio, sr, low_freq, high_freq)
    return (low * low_gain + mid * mid_gain + high * high_gain).astype(np.float32)


def apply_3band_ramp(audio, sr, bands_start, bands_end, low_freq=300, high_freq=3000, power=1.0, hold=0.0):
    """Apply 3-band EQ with per-band gain ramping over the audio duration.

    Each band ramps from its start gain to its end gain. The power parameter
    controls the curve shape: power=1 is linear, power>1 stays near start
    values longer then drops fast (good for delayed EQ rolloff).

    The hold parameter (0-1) keeps bands at their start values for that
    fraction of the duration before ramping. hold=0.8 means 80% hold,
    20% ramp.

    Args:
        audio: mono audio array
        sr: sample rate
        bands_start: dict with keys 'low', 'mid', 'high' — start gains
        bands_end: dict with keys 'low', 'mid', 'high' — end gains
        low_freq: low/mid crossover frequency
        high_freq: mid/high crossover frequency
        power: ramp curve power (1=linear, 3=stays high then drops fast)
        hold: fraction of duration to hold at start values (0-1)

    Returns:
        processed audio array
    """
    low, mid, high = split_3band(audio, sr, low_freq, high_freq)
    n = len(audio)

    if hold > 0:
        hold_samples = int(n * hold)
        ramp_samples = n - hold_samples
        t = np.zeros(n, dtype=np.float32)
        if ramp_samples > 0:
            t[hold_samples:] = np.linspace(0.0, 1.0, ramp_samples).astype(np.float32) ** power
    else:
        t = np.linspace(0.0, 1.0, n).astype(np.float32) ** power

    low_ramp = bands_start.get('low', 1.0) + t * (bands_end.get('low', 1.0) - bands_start.get('low', 1.0))
    mid_ramp = bands_start.get('mid', 1.0) + t * (bands_end.get('mid', 1.0) - bands_start.get('mid', 1.0))
    high_ramp = bands_start.get('high', 1.0) + t * (bands_end.get('high', 1.0) - bands_start.get('high', 1.0))

    return (low * low_ramp + mid * mid_ramp + high * high_ramp).astype(np.float32)


def apply_eq_automation(audio, sr, eq_params, n_steps=8):
    """Apply EQ that changes over the duration of the audio.

    Splits audio into n_steps chunks, applies progressively stronger EQ.
    Useful for transitions where you want EQ to ramp (e.g., gradually cut lows
    on the outgoing track).

    Args:
        audio: mono audio array
        sr: sample rate
        eq_params: EQ band dicts (same format as apply_eq)
            Include "gain_db_start" and "gain_db_end" for ramping.
        n_steps: number of chunks to process

    Returns:
        processed audio array
    """
    chunk_size = len(audio) // n_steps
    output = np.zeros_like(audio)

    for i in range(n_steps):
        start = i * chunk_size
        end = start + chunk_size if i < n_steps - 1 else len(audio)
        chunk = audio[start:end]
        t = i / (n_steps - 1) if n_steps > 1 else 1.0

        # Interpolate EQ parameters for this chunk
        step_params = []
        for band in eq_params:
            step_band = dict(band)
            if "gain_db_start" in band and "gain_db_end" in band:
                step_band["gain_db"] = (
                    band["gain_db_start"] + t * (band["gain_db_end"] - band["gain_db_start"])
                )
            step_params.append(step_band)

        output[start:end] = apply_eq(chunk, sr, step_params)

    return output
