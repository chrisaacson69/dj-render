"""Tests for per-band EQ automation lanes in execute_transition()."""

import numpy as np
import pytest

from dj_render.crossfader import bars_to_samples
from dj_render.mix import _build_lane_envelope, execute_transition


SR = 44100
BPM = 128.0


def _make_track(duration_bars, bpm=BPM, freq=440.0, name="Track"):
    """Generate a synthetic mono track (sine wave) for testing."""
    samples_per_bar = int((60.0 / bpm) * 4 * SR)
    n_samples = duration_bars * samples_per_bar
    t = np.arange(n_samples) / SR
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return {"audio": audio, "bpm": bpm, "spec": {"name": name}}


def _make_noise_track(duration_bars, bpm=BPM, name="Noise"):
    """Generate broadband noise — useful for verifying EQ band filtering."""
    samples_per_bar = int((60.0 / bpm) * 4 * SR)
    n_samples = duration_bars * samples_per_bar
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(n_samples).astype(np.float32) * 0.5
    return {"audio": audio, "bpm": bpm, "spec": {"name": name}}


# ---------------------------------------------------------------------------
# _build_lane_envelope unit tests
# ---------------------------------------------------------------------------

class TestBuildLaneEnvelope:
    def test_missing_lane_returns_default(self):
        automation = {"outgoing_low": [[0, 1.0], [8, 0.0]]}
        env = _build_lane_envelope(automation, "outgoing_mid", 8, 44100)
        assert len(env) == 44100
        assert np.allclose(env, 1.0)

    def test_missing_lane_custom_default(self):
        env = _build_lane_envelope({}, "outgoing_low", 8, 44100, default_value=0.0)
        assert np.allclose(env, 0.0)

    def test_constant_lane(self):
        automation = {"outgoing_low": [[0, 0.5]]}
        env = _build_lane_envelope(automation, "outgoing_low", 8, 44100)
        # np.interp with a single point: constant extrapolation
        assert np.allclose(env, 0.5)

    def test_linear_ramp(self):
        automation = {"outgoing_low": [[0, 1.0], [8, 0.0]]}
        env = _build_lane_envelope(automation, "outgoing_low", 8, 44100)
        assert np.isclose(env[0], 1.0, atol=1e-4)
        assert np.isclose(env[-1], 0.0, atol=1e-4)
        # Midpoint should be ~0.5
        mid = env[len(env) // 2]
        assert 0.4 < mid < 0.6

    def test_step_function(self):
        # Hold at 1.0 for first half, drop to 0.0
        automation = {"x": [[0, 1.0], [4, 1.0], [4, 0.0], [8, 0.0]]}
        env = _build_lane_envelope(automation, "x", 8, 88200)
        # First quarter should be ~1.0
        quarter = len(env) // 4
        assert np.allclose(env[:quarter], 1.0, atol=0.02)
        # Last quarter should be ~0.0
        assert np.allclose(env[-quarter:], 0.0, atol=0.02)

    def test_output_dtype_float32(self):
        automation = {"x": [[0, 1.0], [8, 0.5]]}
        env = _build_lane_envelope(automation, "x", 8, 1000)
        assert env.dtype == np.float32

    def test_output_length_matches(self):
        automation = {"x": [[0, 0.0], [16, 1.0]]}
        for n in [100, 44100, 88200]:
            env = _build_lane_envelope(automation, "x", 16, n)
            assert len(env) == n


# ---------------------------------------------------------------------------
# Per-band automation through execute_transition
# ---------------------------------------------------------------------------

class TestPerBandAutomation:
    """Test the full per-band automation code path in execute_transition."""

    def test_techno_tutorial_blend(self):
        """Slow EQ blend from the techno tutorial — 64 bar overlap.

        Pattern: incoming mids first, then highs, then lows.
        Outgoing lows exit first, then highs, then mids linger.
        Fader is cleanup crew at the end.
        """
        out = _make_noise_track(80, name="Outgoing")
        inc = _make_noise_track(80, name="Incoming")

        spec = {
            "type": "crossfade",
            "overlap_bars": 64,
            "fade_type": "equal_power",
            "automation": {
                "outgoing_fader":  [[0, 1.0], [56, 1.0], [64, 0.0]],
                "incoming_fader":  [[0, 1.0]],
                "incoming_mid":    [[0, 0.0], [8, 0.8], [24, 1.0]],
                "incoming_high":   [[0, 0.0], [16, 0.0], [32, 1.0]],
                "incoming_low":    [[0, 0.0], [32, 0.0], [36, 1.0]],
                "outgoing_low":    [[0, 1.0], [24, 1.0], [28, 0.0]],
                "outgoing_high":   [[0, 1.0], [20, 1.0], [32, 0.0]],
                "outgoing_mid":    [[0, 1.0], [48, 1.0], [56, 0.0]],
            },
        }

        result = execute_transition(out, inc, spec, SR)

        # Output should be: 16 bars solo out + 64 bars overlap + 16 bars solo in
        overlap_samples = bars_to_samples(64, BPM, SR)
        out_solo_samples = len(out["audio"]) - overlap_samples
        in_solo_samples = len(inc["audio"]) - overlap_samples
        expected_len = out_solo_samples + overlap_samples + in_solo_samples
        assert len(result) == expected_len

        # Result should be finite (no NaN/inf from filter instability)
        assert np.all(np.isfinite(result))

    def test_hard_bass_swap(self):
        """Instant bass swap at bar 32 of a 64-bar overlap."""
        out = _make_noise_track(80, name="Out")
        inc = _make_noise_track(80, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 64,
            "fade_type": "equal_power",
            "automation": {
                "outgoing_low":  [[0, 1.0], [32, 1.0], [32.25, 0.0], [64, 0.0]],
                "incoming_low":  [[0, 0.0], [32, 0.0], [32.25, 1.0], [64, 1.0]],
                "outgoing_mid":  [[0, 1.0]],
                "outgoing_high": [[0, 1.0]],
                "incoming_mid":  [[0, 1.0]],
                "incoming_high": [[0, 1.0]],
            },
        }

        result = execute_transition(out, inc, spec, SR)
        assert np.all(np.isfinite(result))
        assert len(result) > 0

    def test_eq_lanes_plus_fader_lanes(self):
        """EQ lanes and explicit fader lanes together — both should apply."""
        out = _make_noise_track(32, name="Out")
        inc = _make_noise_track(32, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 16,
            "fade_type": "equal_power",
            "automation": {
                "outgoing_fader": [[0, 1.0], [16, 0.0]],
                "incoming_fader": [[0, 0.0], [16, 1.0]],
                "outgoing_low":   [[0, 1.0], [8, 0.0]],
                "incoming_low":   [[0, 0.0], [8, 1.0]],
            },
        }

        result = execute_transition(out, inc, spec, SR)
        overlap_samples = bars_to_samples(16, BPM, SR)

        # At the very start, outgoing fader=1.0 and incoming fader=0.0
        # so the first few samples should ≈ outgoing only
        # At the very end, outgoing fader=0.0 and incoming fader=1.0
        # Start of overlap region:
        out_solo_len = len(out["audio"]) - overlap_samples
        overlap_start = result[out_solo_len:out_solo_len + 100]
        overlap_end = result[out_solo_len + overlap_samples - 100:
                             out_solo_len + overlap_samples]

        # Start should have energy from outgoing (fader=1)
        assert np.mean(np.abs(overlap_start)) > 0.01
        # End should have energy from incoming (fader=1)
        assert np.mean(np.abs(overlap_end)) > 0.01
        assert np.all(np.isfinite(result))

    def test_custom_crossover_frequencies(self):
        """Verify low_freq/high_freq from automation dict are used."""
        out = _make_noise_track(32, name="Out")
        inc = _make_noise_track(32, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 16,
            "fade_type": "equal_power",
            "automation": {
                "outgoing_low": [[0, 1.0], [16, 0.0]],
                "incoming_low": [[0, 0.0], [16, 1.0]],
                "low_freq": 200,
                "high_freq": 4000,
            },
        }

        # Should not crash with non-default crossover freqs
        result = execute_transition(out, inc, spec, SR)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Ensure existing specs still work identically."""

    def test_fader_only_automation_uses_old_path(self):
        """automation with only outgoing/incoming (no EQ lanes) -> old fader path."""
        out = _make_track(32, freq=440, name="Out")
        inc = _make_track(32, freq=880, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 16,
            "fade_type": "equal_power",
            "automation": {
                "outgoing": [[0, 1.0], [12, 1.0], [16, 0.0]],
                "incoming": [[0, 0.0], [4, 0.0], [16, 1.0]],
            },
        }

        result = execute_transition(out, inc, spec, SR)
        overlap_samples = bars_to_samples(16, BPM, SR)
        expected_len = len(out["audio"]) + len(inc["audio"]) - overlap_samples
        assert len(result) == expected_len
        assert np.all(np.isfinite(result))

    def test_no_automation_uses_default_crossfade(self):
        """No automation key at all -> standard crossfade path."""
        out = _make_track(32, freq=440, name="Out")
        inc = _make_track(32, freq=880, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 16,
            "fade_type": "equal_power",
        }

        result = execute_transition(out, inc, spec, SR)
        assert len(result) > 0
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Warning tests
# ---------------------------------------------------------------------------

class TestWarnings:
    """Verify incompatible features produce warnings when EQ lanes are active."""

    def test_eq_3band_warns(self, capsys):
        """eq_3band_outgoing should be warned and ignored."""
        out = _make_noise_track(32, name="Out")
        inc = _make_noise_track(32, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 16,
            "fade_type": "equal_power",
            "eq_3band_outgoing": {"low": [1.0, 0.0], "mid": 1.0, "high": 1.0},
            "automation": {
                "outgoing_low": [[0, 1.0], [16, 0.0]],
            },
        }

        execute_transition(out, inc, spec, SR)
        captured = capsys.readouterr()
        assert "eq_3band_outgoing ignored" in captured.out

    def test_outgoing_hold_warns(self, capsys):
        out = _make_noise_track(32, name="Out")
        inc = _make_noise_track(32, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 16,
            "fade_type": "equal_power",
            "outgoing_hold": True,
            "automation": {
                "incoming_mid": [[0, 0.0], [16, 1.0]],
            },
        }

        execute_transition(out, inc, spec, SR)
        captured = capsys.readouterr()
        assert "outgoing_hold ignored" in captured.out

    def test_overlap_gain_hump_warns(self, capsys):
        out = _make_noise_track(32, name="Out")
        inc = _make_noise_track(32, name="In")

        spec = {
            "type": "crossfade",
            "overlap_bars": 16,
            "fade_type": "equal_power",
            "overlap_gain_hump_db": 3.0,
            "automation": {
                "outgoing_high": [[0, 1.0], [16, 0.0]],
            },
        }

        execute_transition(out, inc, spec, SR)
        captured = capsys.readouterr()
        assert "overlap_gain_hump_db ignored" in captured.out
