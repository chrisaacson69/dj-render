"""Time-stretching: change BPM without changing pitch.

Uses pyrubberband (the Rubber Band library) for high-quality pitch-preserving
time-stretch with transient preservation. Falls back to ffmpeg atempo if
pyrubberband is not available.
"""

import subprocess
import tempfile
import os

import numpy as np
import soundfile as sf

try:
    import pyrubberband as pyrb
    _HAS_RUBBERBAND = True
except ImportError:
    _HAS_RUBBERBAND = False


def time_stretch(audio, sr, original_bpm, target_bpm):
    """Stretch audio to match a target BPM.

    Args:
        audio: mono audio array (float32)
        sr: sample rate
        original_bpm: the track's native BPM
        target_bpm: desired BPM

    Returns:
        stretched audio array
    """
    if abs(original_bpm - target_bpm) < 0.01:
        return audio

    rate = target_bpm / original_bpm

    if _HAS_RUBBERBAND:
        return _stretch_rubberband(audio, sr, rate)
    else:
        return _stretch_ffmpeg(audio, sr, rate)


def _stretch_rubberband(audio, sr, rate):
    """Time-stretch using pyrubberband (Rubber Band library).

    Rubber Band is a phase vocoder with transient preservation — the same
    engine used by Ableton Live's Complex Pro warp mode. It preserves
    timing accuracy and audio quality far better than ffmpeg's atempo.
    """
    stretched = pyrb.time_stretch(audio, sr, rate)
    return stretched.astype(np.float32)


def _stretch_ffmpeg(audio, sr, rate):
    """Fallback: time-stretch using ffmpeg's atempo filter.

    ffmpeg's atempo filter changes tempo without changing pitch.
    Lower quality than rubberband — introduces timing drift on longer files.
    atempo is limited to [0.5, 100.0] range; chain multiple filters if needed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.wav")
        out_path = os.path.join(tmpdir, "out.wav")

        # Write input
        sf.write(in_path, audio, sr)

        # Build atempo filter chain (atempo range is 0.5 to 100.0)
        # For rates outside [0.5, 2.0], chain multiple atempo filters
        atempo_chain = _build_atempo_chain(rate)

        cmd = [
            "ffmpeg", "-i", in_path,
            "-filter:a", atempo_chain,
            "-v", "quiet",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg time-stretch failed: {result.stderr.decode()}"
            )

        # Read output
        stretched, _ = sf.read(out_path, dtype="float32")
        if stretched.ndim == 2:
            stretched = np.mean(stretched, axis=1)
        return stretched


def _build_atempo_chain(rate):
    """Build ffmpeg atempo filter chain for the given rate.

    atempo accepts values in [0.5, 100.0]. For values outside [0.5, 2.0],
    we chain multiple filters for better quality.
    """
    if 0.5 <= rate <= 2.0:
        return f"atempo={rate}"

    # Chain filters to stay within reasonable per-step range
    filters = []
    remaining = rate
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining}")
    return ",".join(filters)


def recompute_bars_after_stretch(bar_frames, original_bpm, target_bpm):
    """Adjust bar frame positions after time-stretching.

    Time-stretching changes the duration, so bar positions shift proportionally.

    Args:
        bar_frames: original bar positions in samples
        original_bpm: BPM before stretching
        target_bpm: BPM after stretching

    Returns:
        adjusted bar_frames array
    """
    if abs(original_bpm - target_bpm) < 0.01:
        return bar_frames

    rate = target_bpm / original_bpm
    return (bar_frames / rate).astype(int)
