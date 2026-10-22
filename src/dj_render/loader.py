"""Audio loading, first-downbeat detection, and bar grid computation.

Uses soundfile for WAV/FLAC and ffmpeg subprocess for MP3.
Bar grid anchored at the detected first downbeat using scipy-only onset detection.
"""

import os
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, find_peaks


def load_track(file_path, target_sr=44100):
    """Load an audio file and return as mono float32 numpy array.

    Supports WAV, FLAC (via soundfile) and MP3 (via ffmpeg subprocess).

    Returns:
        tuple: (audio_array, sample_rate)
    """
    if file_path.lower().endswith('.mp3'):
        return _load_mp3_ffmpeg(file_path, target_sr)

    audio, sr = sf.read(file_path, dtype="float32")

    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def _load_mp3_ffmpeg(file_path, target_sr):
    """Decode MP3 to float32 mono using ffmpeg subprocess."""
    cmd = [
        "ffmpeg", "-i", file_path,
        "-f", "f32le",         # 32-bit float, little-endian raw PCM
        "-acodec", "pcm_f32le",
        "-ar", str(target_sr), # resample to target
        "-ac", "1",            # mono
        "-v", "quiet",
        "pipe:1",              # output to stdout
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {file_path}: {result.stderr.decode()}"
        )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio, target_sr


def _resample(audio, orig_sr, target_sr):
    """Simple resampling using numpy interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    x_old = np.linspace(0, duration, len(audio))
    x_new = np.linspace(0, duration, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# --- First downbeat detection ---

def detect_first_downbeat(audio, sr, window_ms=50.0, threshold_ratio=0.20):
    """Detect the sample position of the first significant downbeat.

    Uses a bandpass filter to isolate kick drum frequencies (20-150 Hz),
    then finds the first significant energy peak in the RMS envelope.

    Args:
        audio: mono float32 audio array
        sr: sample rate
        window_ms: RMS window size in ms
        threshold_ratio: fraction of max energy that counts as a beat

    Returns:
        sample position of the first detected downbeat (0 if not found)
    """
    audio = np.asarray(audio, dtype=np.float32)
    n_samples = len(audio)

    # Bandpass filter: isolate kick drum frequencies (20-150 Hz)
    nyquist = sr / 2.0
    lo = max(20.0, 1.0) / nyquist
    hi = min(150.0, nyquist - 1.0) / nyquist
    if lo < hi:
        sos = butter(4, [lo, hi], btype='band', output='sos')
        filtered = sosfilt(sos, audio)
    else:
        filtered = audio

    # Short-time RMS energy envelope
    win_samples = max(int(sr * window_ms / 1000.0), 1)
    hop = max(win_samples // 2, 1)
    n_frames = (n_samples - win_samples) // hop + 1
    if n_frames <= 0:
        return 0

    envelope = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        window = filtered[start:start + win_samples]
        envelope[i] = np.sqrt(np.mean(window * window))

    max_energy = np.max(envelope)
    if max_energy < 1e-10:
        return 0

    threshold = threshold_ratio * max_energy

    # Find peaks with minimum distance (assume fastest kick ~200 BPM = 0.15s apart)
    min_dist_frames = max(int(0.15 * sr / hop), 1)
    min_skip_frames = max(int(0.005 * sr / hop), 0)  # skip first 5ms

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=min_dist_frames,
        prominence=threshold * 0.3,
    )
    peaks = peaks[peaks >= min_skip_frames]

    if len(peaks) == 0:
        above = np.where(envelope >= threshold)[0]
        above = above[above >= min_skip_frames]
        if len(above) == 0:
            return 0
        first_frame = above[0]
    else:
        first_frame = peaks[0]

    # Walk back to the onset of the transient
    onset_threshold = envelope[first_frame] * 0.10
    onset_frame = first_frame
    while onset_frame > 0 and envelope[onset_frame - 1] > onset_threshold:
        onset_frame -= 1

    return int(onset_frame * hop)


# --- Bar grid computation ---

def compute_bar_frames(bpm, sr, total_samples, first_beat_sample=0, beats_per_bar=4):
    """Compute bar positions phase-aligned to the first downbeat.

    The detected first downbeat sets the phase (where beats land), but
    bar numbering extends backwards to sample 0 so that bar numbers
    are relative to the start of the audio file.

    Example: if first_beat is at 15.8s and bars are 1.95s each,
    there are ~8 bars of quiet intro. Bar 0 starts at the phase-correct
    position near sample 0, and bar 8 lands exactly on the first kick.

    Args:
        bpm: known BPM of the track
        sr: sample rate
        total_samples: length of audio in samples
        first_beat_sample: sample position of the first downbeat
        beats_per_bar: beats per bar (default 4 for 4/4 time)

    Returns:
        bar_frames: array of bar start positions in samples
    """
    seconds_per_bar = (60.0 / bpm) * beats_per_bar
    samples_per_bar = seconds_per_bar * sr

    # Extrapolate backwards: find the phase-correct bar 0 position
    # that's at or just after sample 0
    if first_beat_sample > 0:
        bars_before = int(first_beat_sample / samples_per_bar)
        bar_0_offset = first_beat_sample - bars_before * samples_per_bar
    else:
        bar_0_offset = 0

    n_bars = int((total_samples - bar_0_offset) / samples_per_bar) + 1
    bar_frames = np.array([
        int(bar_0_offset + i * samples_per_bar) for i in range(n_bars)
    ])
    return bar_frames


def bar_to_sample(bar_number, bar_frames):
    """Convert a bar number to a sample position.

    Supports fractional bars: 199.75 = bar 199 + 75% of bar duration
    (i.e., beat 4 in 4/4 time).

    Args:
        bar_number: which bar (0-indexed), can be fractional
        bar_frames: array of bar start positions in samples

    Returns:
        sample position, or None if bar_number is out of range
    """
    int_bar = int(bar_number)
    frac = bar_number - int_bar

    if int_bar < 0 or int_bar >= len(bar_frames):
        return None

    sample = bar_frames[int_bar]
    if frac > 0 and int_bar + 1 < len(bar_frames):
        next_sample = bar_frames[int_bar + 1]
        sample = sample + frac * (next_sample - sample)

    return int(sample)


def trim_to_bars(audio, bar_frames, start_bar, end_bar):
    """Extract audio between two bar positions.

    Args:
        audio: full audio array
        bar_frames: array of bar start positions in samples
        start_bar: first bar (inclusive)
        end_bar: last bar (exclusive)

    Returns:
        trimmed audio array
    """
    start_sample = bar_to_sample(start_bar, bar_frames)
    end_sample = bar_to_sample(end_bar, bar_frames)

    if start_sample is None:
        start_sample = 0
    if end_sample is None:
        end_sample = len(audio)

    return audio[start_sample:end_sample]


def apply_loop(audio, bar_frames, loop_spec, position="start", sr=44100,
               crossfade_samples=1024):
    """Loop the first or last N bars of a track to extend intro/outro runway.

    Args:
        audio: mono float32 audio array (already trimmed to cue points)
        bar_frames: array of bar start positions in samples (rebased to trimmed audio)
        loop_spec: dict with "bars" (how many bars to loop) and "copies" (total repetitions)
        position: "start" (loop first N bars) or "end" (loop last N bars)
        sr: sample rate
        crossfade_samples: micro-crossfade length at each seam (~23ms at 44100)

    Returns:
        tuple: (new_audio, new_bar_frames) with looped content stitched in
    """
    loop_bars = loop_spec["bars"]
    copies = loop_spec["copies"]

    if copies <= 1:
        return audio, bar_frames

    total_bars = len(bar_frames)
    if loop_bars >= total_bars:
        print(f"    WARNING: loop_bars ({loop_bars}) >= total bars ({total_bars}), skipping loop")
        return audio, bar_frames

    if position == "start":
        # Extract first N bars
        loop_end_sample = int(bar_frames[loop_bars]) if loop_bars < len(bar_frames) else len(audio)
        loop_region = audio[:loop_end_sample]
        remainder = audio[loop_end_sample:]
    else:
        # Extract last N bars
        loop_start_bar = total_bars - loop_bars
        loop_start_sample = int(bar_frames[loop_start_bar])
        loop_region = audio[loop_start_sample:]
        remainder = audio[:loop_start_sample]

    # Tile the loop region
    region_len = len(loop_region)
    xf = min(crossfade_samples, region_len // 4)  # don't crossfade more than 25% of region

    tiles = []
    for c in range(copies):
        tile = loop_region.copy()
        if c > 0 and xf > 0:
            # Equal-power micro-crossfade at seam: blend end of previous tile with start of this one
            t = np.linspace(0, np.pi / 2, xf).astype(np.float32)
            fade_in = np.sin(t)
            fade_out = np.cos(t)
            # Fade out the tail of the previous tile
            tiles[-1][-xf:] *= fade_out
            # Fade in the head of this tile
            tile[:xf] *= fade_in
            # Overlap-add at the seam
            tiles[-1][-xf:] += tile[:xf]
            tile = tile[xf:]  # trim the already-blended head
        tiles.append(tile)

    looped = np.concatenate(tiles)

    # Compute bar duration from the loop region
    if loop_bars > 0 and loop_bars < len(bar_frames):
        samples_per_bar = (bar_frames[min(loop_bars, len(bar_frames) - 1)] - bar_frames[0]) / loop_bars
    else:
        samples_per_bar = region_len / max(loop_bars, 1)

    # Build new bar_frames
    looped_bar_count = loop_bars * copies
    extra_bars = loop_bars * (copies - 1)  # how many bars we added

    if position == "start":
        # New bar frames: looped bars + shifted remainder bars
        new_bars = np.array([int(i * samples_per_bar) for i in range(looped_bar_count)],
                            dtype=bar_frames.dtype)
        # Shift remainder bar_frames forward by the extension amount
        extension = len(looped) - region_len
        remainder_bars = bar_frames[loop_bars:] + extension
        new_bar_frames = np.concatenate([new_bars, remainder_bars])
        new_audio = np.concatenate([looped, remainder])
    else:
        # New bar frames: remainder bars + looped bars
        loop_start_bar = total_bars - loop_bars
        remainder_bars = bar_frames[:loop_start_bar].copy()
        loop_offset = int(remainder_bars[-1]) if len(remainder_bars) > 0 else 0
        # Add bar spacing for the looped section
        if len(remainder_bars) > 1:
            loop_offset = int(remainder_bars[-1])
            bar_gap = int(remainder_bars[-1] - remainder_bars[-2]) if len(remainder_bars) > 1 else int(samples_per_bar)
            loop_offset += bar_gap
        else:
            loop_offset = len(remainder)
        new_loop_bars = np.array([int(loop_offset + i * samples_per_bar) for i in range(looped_bar_count)],
                                 dtype=bar_frames.dtype)
        new_bar_frames = np.concatenate([remainder_bars, new_loop_bars])
        new_audio = np.concatenate([remainder, looped])

    loop_dur = len(looped) / sr
    print(f"    Loop ({position}): {loop_bars} bars x {copies} copies = {looped_bar_count} bars ({loop_dur:.1f}s)")

    return new_audio, new_bar_frames


def load_and_prepare(track_spec, tracks_dir, target_sr=44100):
    """Load a track, detect its first downbeat, and compute its bar grid.

    Args:
        track_spec: dict from mix spec with file, original_bpm, cue_in_bar, cue_out_bar
        tracks_dir: path to directory containing audio files
        target_sr: target sample rate

    Returns:
        dict with keys: audio, sr, bpm, bar_frames, first_beat, spec
    """
    file_path = os.path.join(tracks_dir, track_spec["file"])
    audio, sr = load_track(file_path, target_sr)

    bpm = track_spec["original_bpm"]

    # Detect first downbeat (or use manual override from spec)
    first_beat = track_spec.get("first_beat_sample")
    if first_beat is None:
        first_beat = detect_first_downbeat(audio, sr)

    bar_frames = compute_bar_frames(bpm, sr, len(audio), first_beat)

    return {
        "audio": audio,
        "sr": sr,
        "bpm": bpm,
        "bar_frames": bar_frames,
        "first_beat": first_beat,
        "spec": track_spec,
    }
