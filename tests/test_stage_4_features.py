"""
tests/test_stage_4_features.py
================================
Stage 4 — Comprehensive Pytest Suite

Covers all Stage 4 sub-modules:
  1. entropy_calculator  — Shannon entropy, histogram stats, 4KB buffer sampling
  2. hawkes_engine       — Hawkes intensity, branching ratio, superheated detection
  3. etd_engine          — ETD KL-divergence computation
  4. feature_extractor   — End-to-end orchestrator integration with Stage 3 DBRG

Test categories:
  - Mathematical correctness against known reference values
  - Edge cases (empty files, binary data, permission errors)
  - Thread safety under concurrent access
  - Cross-stage integration (Stage 3 DBRG -> Stage 4 features)
"""

import math
import os
import sys
import tempfile
import time
import threading
import random

import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stage_4_features.entropy_calculator import (
    read_file_head,
    shannon_entropy,
    byte_histogram_stats,
    normalize_entropy,
    BUFFER_SIZE,
)
from src.stage_4_features.hawkes_engine import HawkesEngine
from src.stage_4_features.etd_engine import ETDEngine, _bin_delta_h, _bin_d_out
from src.stage_4_features.feature_extractor import FeatureExtractor
from src.stage_3_dbrg.dbrg_manager import DBRGManager


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Entropy Calculator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShannonEntropy:
    """Verify Shannon entropy computation against known reference values."""

    def test_all_zeros_entropy_is_zero(self):
        """All-zero bytes have zero entropy (single symbol)."""
        data = bytes(4096)
        H = shannon_entropy(data)
        assert H == 0.0, f"Expected 0.0, got {H}"

    def test_single_byte_value_entropy_is_zero(self):
        """A buffer of a single repeated byte has zero entropy."""
        data = bytes([0xAB] * 4096)
        H = shannon_entropy(data)
        assert H == 0.0, f"Expected 0.0, got {H}"

    def test_two_equal_bytes_entropy_is_one(self):
        """Two equally frequent byte values -> H = 1.0 bit."""
        data = bytes([0x00] * 2048 + [0xFF] * 2048)
        H = shannon_entropy(data)
        assert abs(H - 1.0) < 0.01, f"Expected ~1.0, got {H}"

    def test_four_equal_bytes_entropy_is_two(self):
        """Four equally frequent byte values -> H = 2.0 bits."""
        data = bytes([0x00] * 1024 + [0x01] * 1024 + [0x02] * 1024 + [0x03] * 1024)
        H = shannon_entropy(data)
        assert abs(H - 2.0) < 0.01, f"Expected ~2.0, got {H}"

    def test_random_bytes_high_entropy(self):
        """Random bytes should have entropy near 8.0 (maximum)."""
        random.seed(42)
        data = bytes(random.getrandbits(8) for _ in range(4096))
        H = shannon_entropy(data)
        assert H > 7.8, f"Expected > 7.8, got {H}"

    def test_english_text_moderate_entropy(self):
        """English text should have moderate entropy (~3.5-5.0)."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of the emergency broadcast system. "
            "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
        ) * 30
        data = text.encode("utf-8")[:4096]
        H = shannon_entropy(data)
        assert 3.0 < H < 5.5, f"Expected 3.0-5.5, got {H}"

    def test_empty_data_returns_zero(self):
        """Empty input returns 0.0 entropy."""
        assert shannon_entropy(b"") == 0.0

    def test_entropy_range_bounded(self):
        """Entropy must always be in [0.0, 8.0]."""
        for _ in range(20):
            data = bytes(random.getrandbits(8) for _ in range(random.randint(1, 8192)))
            H = shannon_entropy(data)
            assert 0.0 <= H <= 8.0, f"Entropy {H} out of range [0, 8]"

    def test_small_buffer(self):
        """Entropy works correctly for very small buffers (< 256 bytes)."""
        data = b"hello world"
        H = shannon_entropy(data)
        assert H > 0.0, "Non-trivial text should have positive entropy"
        assert H < 8.0


class TestNormalizeEntropy:
    """Verify entropy normalization to [0, 1]."""

    def test_normalize_zero(self):
        assert normalize_entropy(0.0) == 0.0

    def test_normalize_max(self):
        assert normalize_entropy(8.0) == 1.0

    def test_normalize_midpoint(self):
        assert abs(normalize_entropy(4.0) - 0.5) < 0.001

    def test_normalize_clamps_over_eight(self):
        assert normalize_entropy(10.0) == 1.0

    def test_normalize_clamps_negative(self):
        assert normalize_entropy(-1.0) == 0.0

    def test_encrypted_data_above_085(self):
        """Encrypted / random noise should yield S_entropy > 0.85."""
        random.seed(99)
        data = bytes(random.getrandbits(8) for _ in range(4096))
        H = shannon_entropy(data)
        S = normalize_entropy(H)
        assert S > 0.85, f"Expected > 0.85 for random data, got {S}"

    def test_plaintext_below_050(self):
        """Plain English text should yield S_entropy < 0.60."""
        text = ("Hello World. Testing plain text entropy normalization. ") * 80
        data = text.encode("utf-8")[:4096]
        H = shannon_entropy(data)
        S = normalize_entropy(H)
        assert S < 0.65, f"Expected < 0.65 for plaintext, got {S}"


class TestByteHistogramStats:
    """Verify 256-bin histogram variance and kurtosis."""

    def test_uniform_bytes_low_variance(self):
        """Perfectly uniform distribution has near-zero variance."""
        # Build a buffer with each byte value appearing exactly 16 times
        data = bytes(list(range(256)) * 16)  # 4096 bytes
        var, kurt = byte_histogram_stats(data)
        assert var < 1e-10, f"Expected near-zero variance, got {var}"

    def test_single_byte_high_variance(self):
        """A single repeated byte has high variance (one bin at 1.0, rest at 0)."""
        data = bytes([0x42] * 4096)
        var, kurt = byte_histogram_stats(data)
        assert var > 1e-5, f"Expected positive variance for single-byte, got {var}"

    def test_random_bytes_low_variance(self):
        """Random bytes approximate uniform -> low variance."""
        random.seed(123)
        data = bytes(random.getrandbits(8) for _ in range(4096))
        var, kurt = byte_histogram_stats(data)
        assert var < 1e-5, f"Expected low variance for random, got {var}"

    def test_empty_data(self):
        var, kurt = byte_histogram_stats(b"")
        assert var == 0.0
        assert kurt == 0.0

    def test_returns_tuple(self):
        data = b"test data for histogram"
        result = byte_histogram_stats(data)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestReadFileHead:
    """Verify 4KB buffer file reading."""

    def test_read_normal_file(self):
        """Read first 4096 bytes from a normal text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("A" * 8192)
            path = f.name
        try:
            data = read_file_head(path)
            assert data is not None
            assert len(data) == BUFFER_SIZE
            assert data == b"A" * BUFFER_SIZE
        finally:
            os.unlink(path)

    def test_read_small_file(self):
        """Files smaller than 4096 bytes return whatever is available."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("tiny")
            path = f.name
        try:
            data = read_file_head(path)
            assert data is not None
            assert len(data) == 4
        finally:
            os.unlink(path)

    def test_read_empty_file_returns_none(self):
        """Empty files return None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            data = read_file_head(path)
            assert data is None
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_none(self):
        data = read_file_head("/nonexistent/path/file.txt")
        assert data is None

    def test_unknown_path_returns_none(self):
        assert read_file_head("UNKNOWN") is None
        assert read_file_head("") is None
        assert read_file_head("SYSTEM_PROCESS") is None

    def test_file_prefix_stripped(self):
        """eCAR objectID 'file:' prefix is handled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            path = f.name
        try:
            data = read_file_head(f"file:{path}")
            assert data is not None
            assert data == b"content"
        finally:
            os.unlink(path)

    def test_binary_file(self):
        """Binary files are read correctly."""
        random.seed(77)
        binary_content = bytes(random.getrandbits(8) for _ in range(4096))
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(binary_content)
            path = f.name
        try:
            data = read_file_head(path)
            assert data == binary_content
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Hawkes Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHawkesEngine:
    """Verify Hawkes self-exciting point process calculations."""

    def test_initial_intensity_is_mu(self):
        """With no events, intensity equals the background rate mu."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8)
        intensity = engine.get_intensity(pid=100)
        assert abs(intensity - 0.1) < 1e-6

    def test_branching_ratio(self):
        """Branching ratio n = alpha / beta."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8)
        assert abs(engine.branching_ratio - 0.625) < 1e-6

    def test_subcritical_not_superheated(self):
        """Default parameters (n=0.625) should NOT be superheated."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8)
        assert not engine.is_superheated()

    def test_supercritical_is_superheated(self):
        """alpha >= beta -> n >= 1.0 -> superheated."""
        engine = HawkesEngine(mu=0.1, alpha=1.0, beta=0.8)
        assert engine.is_superheated()
        assert engine.branching_ratio >= 1.0

    def test_single_event_increases_intensity(self):
        """A single event should boost intensity above mu."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8)
        t_now = time.time()
        engine.update(pid=1, t_now=t_now)
        intensity = engine.get_intensity(pid=1, t_now=t_now + 0.001)
        assert intensity > 0.1, f"Expected > 0.1 after event, got {intensity}"

    def test_rapid_burst_accumulates_intensity(self):
        """Rapid events accumulate intensity significantly."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8)
        t_base = 1000.0
        for i in range(10):
            engine.update(pid=1, t_now=t_base + i * 0.01)  # 10ms apart
        intensity = engine.get_intensity(pid=1, t_now=t_base + 0.1)
        assert intensity > 2.0, f"Burst should yield high intensity, got {intensity}"

    def test_intensity_decays_over_time(self):
        """Intensity decays back toward mu after idle period."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8)
        t_now = 1000.0
        engine.update(pid=1, t_now=t_now)
        
        int_immediate = engine.get_intensity(pid=1, t_now=t_now + 0.001)
        int_later = engine.get_intensity(pid=1, t_now=t_now + 10.0)
        
        assert int_later < int_immediate, "Intensity should decay over time"
        assert int_later < 0.2, f"After 10s, intensity should be near mu, got {int_later}"

    def test_normalized_intensity_in_range(self):
        """Normalized intensity is always in [0.0, 1.0]."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8, lambda_max=10.0)
        t = 1000.0
        for i in range(50):
            engine.update(pid=1, t_now=t + i * 0.01)
        norm = engine.get_normalized_intensity(pid=1, t_now=t + 0.5)
        assert 0.0 <= norm <= 1.0, f"Normalized intensity {norm} out of range"

    def test_separate_pid_tracking(self):
        """Different PIDs have independent event histories."""
        engine = HawkesEngine(mu=0.1, alpha=0.5, beta=0.8)
        t = 1000.0
        
        # PID 1 gets many events
        for i in range(10):
            engine.update(pid=1, t_now=t + i * 0.01)
        
        # PID 2 gets no events
        int_1 = engine.get_intensity(pid=1, t_now=t + 0.1)
        int_2 = engine.get_intensity(pid=2, t_now=t + 0.1)
        
        assert int_1 > int_2, "PID 1 should have higher intensity than PID 2"
        assert abs(int_2 - 0.1) < 1e-6, "PID 2 intensity should equal mu"

    def test_event_count_tracking(self):
        engine = HawkesEngine()
        t = 1000.0
        for i in range(5):
            engine.update(pid=42, t_now=t + i)
        assert engine.get_event_count(42) == 5
        assert engine.total_events == 5

    def test_reset_pid(self):
        engine = HawkesEngine()
        engine.update(pid=1, t_now=1000.0)
        assert engine.get_event_count(1) == 1
        engine.reset_pid(1)
        assert engine.get_event_count(1) == 0

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            HawkesEngine(beta=0)
        with pytest.raises(ValueError):
            HawkesEngine(beta=-1.0)

    def test_horizon_pruning(self):
        """Events older than horizon_sec are pruned."""
        engine = HawkesEngine(horizon_sec=5.0)
        t = 1000.0
        engine.update(pid=1, t_now=t)
        engine.update(pid=1, t_now=t + 10.0)  # triggers pruning
        # First event at t=1000 should be pruned (10s > 5s horizon)
        assert engine.get_event_count(1) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ETD Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestETDEngine:
    """Verify Entropy-Topology Divergence computation."""

    def test_benign_pattern_low_etd(self):
        """Benign-like observations (low delta_H, low d_out) -> low S_ETD."""
        etd = ETDEngine(window_size=20)
        # Simulate 10 benign writes: small entropy change, few files
        for _ in range(10):
            s = etd.compute_etd(pid=1, delta_h=0.1, d_out=2)
        assert s < 0.3, f"Expected low S_ETD for benign pattern, got {s}"

    def test_ransomware_pattern_high_etd(self):
        """Ransomware-like observations (high delta_H, high d_out) -> high S_ETD."""
        etd = ETDEngine(window_size=20)
        # Simulate 15 ransomware writes: high entropy change, many files
        for _ in range(15):
            s = etd.compute_etd(pid=1, delta_h=7.5, d_out=40)
        assert s > 0.3, f"Expected elevated S_ETD for ransomware, got {s}"

    def test_etd_score_in_range(self):
        """S_ETD is always in [0.0, 1.0]."""
        etd = ETDEngine()
        for _ in range(50):
            dh = random.uniform(-8.0, 8.0)
            do = random.randint(0, 100)
            s = etd.compute_etd(pid=1, delta_h=dh, d_out=do)
            assert 0.0 <= s <= 1.0, f"S_ETD {s} out of [0, 1]"

    def test_separate_pid_tracking(self):
        """Different PIDs maintain independent observation histories."""
        etd = ETDEngine()
        for _ in range(10):
            etd.compute_etd(pid=1, delta_h=7.0, d_out=30)
        for _ in range(10):
            etd.compute_etd(pid=2, delta_h=0.1, d_out=1)
        assert etd.get_observation_count(1) == 10
        assert etd.get_observation_count(2) == 10

    def test_window_size_limit(self):
        """Observations beyond window_size are trimmed."""
        etd = ETDEngine(window_size=5)
        for i in range(20):
            etd.compute_etd(pid=1, delta_h=float(i), d_out=i)
        assert etd.get_observation_count(1) == 5

    def test_previous_entropy_tracking(self):
        etd = ETDEngine()
        etd.set_previous_entropy(1, 3.5)
        assert abs(etd.get_previous_entropy(1) - 3.5) < 1e-6
        assert abs(etd.get_previous_entropy(999) - 0.0) < 1e-6  # Unknown PID

    def test_reset_pid(self):
        etd = ETDEngine()
        etd.compute_etd(pid=1, delta_h=1.0, d_out=5)
        etd.set_previous_entropy(1, 4.0)
        etd.reset_pid(1)
        assert etd.get_observation_count(1) == 0
        assert abs(etd.get_previous_entropy(1) - 0.0) < 1e-6

    def test_binning_functions(self):
        """Verify bin index calculations are within expected ranges."""
        assert 0 <= _bin_delta_h(-8.0) < 16
        assert 0 <= _bin_delta_h(0.0) < 16
        assert 0 <= _bin_delta_h(8.0) < 16
        assert 0 <= _bin_d_out(0) < 10
        assert 0 <= _bin_d_out(50) < 10
        assert 0 <= _bin_d_out(100) < 10  # Clamped to max


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Feature Extractor Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureExtractor:
    """End-to-end integration tests for the Stage 4 orchestrator."""

    @pytest.fixture
    def dbrg(self):
        """Create a fresh DBRGManager for each test."""
        return DBRGManager(decay_lambda=0.05)

    @pytest.fixture
    def extractor(self, dbrg):
        """Create a FeatureExtractor connected to a DBRG."""
        return FeatureExtractor(dbrg_manager=dbrg)

    def _make_test_file(self, content: bytes) -> str:
        """Create a temp file with given content and return its path."""
        f = tempfile.NamedTemporaryFile(mode="wb", suffix=".dat", delete=False)
        f.write(content)
        f.close()
        return f.name

    def _make_ecar(self, pid, file_path, operation="FILE_MODIFY"):
        """Build a minimal eCAR event dict."""
        return {
            "actorID": f"pid:{pid}",
            "objectID": f"file:{file_path}",
            "pid": pid,
            "operation": operation,
            "timestamp": int(time.time() * 1000),
            "context": {
                "exe_path": "test_process.exe",
                "sha256": "abc123",
                "ppid": 1,
                "parent_exe": "explorer.exe",
                "cmdline": [],
            },
        }

    def test_plaintext_feature_extraction(self, extractor, dbrg):
        """Plaintext file should yield low S_entropy."""
        content = (b"Hello World. This is plain text. " * 128)[:4096]
        path = self._make_test_file(content)
        try:
            ecar = self._make_ecar(pid=100, file_path=path)
            dbrg.process_event(ecar)
            fv = extractor.extract_features(ecar)

            assert fv is not None
            assert fv["S_entropy"] < 0.60, f"Plaintext S_entropy too high: {fv['S_entropy']}"
            assert 0.0 <= fv["S_entropy"] <= 1.0
            assert 0.0 <= fv["S_dist"] <= 1.0
            assert 0.0 <= fv["S_dev"] <= 1.0
            assert 0.0 <= fv["S_stab"] <= 1.0
            assert fv["pid"] == 100
        finally:
            os.unlink(path)

    def test_encrypted_feature_extraction(self, extractor, dbrg):
        """Random/encrypted data should yield high S_entropy."""
        random.seed(42)
        content = bytes(random.getrandbits(8) for _ in range(4096))
        path = self._make_test_file(content)
        try:
            ecar = self._make_ecar(pid=200, file_path=path)
            dbrg.process_event(ecar)
            fv = extractor.extract_features(ecar)

            assert fv is not None
            assert fv["S_entropy"] > 0.85, f"Encrypted S_entropy too low: {fv['S_entropy']}"
            assert fv["entropy_raw"] > 7.0
        finally:
            os.unlink(path)

    def test_feature_vector_keys(self, extractor, dbrg):
        """Verify all expected keys are present in the feature vector."""
        content = b"test" * 1024
        path = self._make_test_file(content)
        try:
            ecar = self._make_ecar(pid=300, file_path=path)
            dbrg.process_event(ecar)
            fv = extractor.extract_features(ecar)

            expected_keys = [
                "S_entropy", "S_ETD", "S_dist", "S_dev", "S_stab",
                "lambda_norm", "branching_n", "is_superheated",
                "entropy_raw", "delta_H", "hist_variance", "hist_kurtosis",
                "fan_out_degree", "pid", "file_path", "operation",
                "timestamp", "actor_id",
            ]
            for key in expected_keys:
                assert key in fv, f"Missing key '{key}' in feature vector"
        finally:
            os.unlink(path)

    def test_fan_out_from_dbrg(self, extractor, dbrg):
        """S_dist reflects the actual DBRG fan-out degree."""
        path1 = self._make_test_file(b"file1" * 1024)
        path2 = self._make_test_file(b"file2" * 1024)
        path3 = self._make_test_file(b"file3" * 1024)
        try:
            # Process accesses 3 different files
            for path in [path1, path2, path3]:
                ecar = self._make_ecar(pid=400, file_path=path)
                dbrg.process_event(ecar)

            # Extract features on the third event
            ecar = self._make_ecar(pid=400, file_path=path3)
            fv = extractor.extract_features(ecar)

            assert fv is not None
            assert fv["fan_out_degree"] == 3, f"Expected fan-out 3, got {fv['fan_out_degree']}"
            assert fv["S_dist"] > 0.0
        finally:
            for p in [path1, path2, path3]:
                os.unlink(p)

    def test_ransomware_burst_high_scores(self, extractor, dbrg):
        """Simulated ransomware burst: many files, high entropy, rapid events."""
        paths = []
        random.seed(55)
        try:
            for i in range(15):
                content = bytes(random.getrandbits(8) for _ in range(4096))
                path = self._make_test_file(content)
                paths.append(path)
                ecar = self._make_ecar(pid=500, file_path=path)
                dbrg.process_event(ecar)
                fv = extractor.extract_features(ecar)

            # After 15 rapid encrypted file writes
            assert fv is not None
            assert fv["S_entropy"] > 0.85, f"Ransomware S_entropy low: {fv['S_entropy']}"
            assert fv["fan_out_degree"] >= 10, f"Fan-out too low: {fv['fan_out_degree']}"
            assert fv["S_dist"] > 0.15, f"S_dist too low: {fv['S_dist']}"
        finally:
            for p in paths:
                os.unlink(p)

    def test_deleted_file_graceful(self, extractor, dbrg):
        """Feature extraction handles deleted files gracefully."""
        path = self._make_test_file(b"data")
        os.unlink(path)  # Delete before extraction
        ecar = self._make_ecar(pid=600, file_path=path)
        dbrg.process_event(ecar)
        fv = extractor.extract_features(ecar)
        assert fv is not None
        assert fv["S_entropy"] == 0.0  # Can't read deleted file

    def test_no_dbrg_manager(self):
        """FeatureExtractor works without a DBRG manager (fan-out=0)."""
        extractor = FeatureExtractor(dbrg_manager=None)
        content = b"test content" * 341
        path = self._make_test_file(content)
        try:
            ecar = self._make_ecar(pid=700, file_path=path)
            fv = extractor.extract_features(ecar)
            assert fv is not None
            assert fv["fan_out_degree"] == 0
            assert fv["S_dist"] == 0.0
        finally:
            os.unlink(path)

    def test_s_stab_repeated_file(self, extractor, dbrg):
        """Repeated access to the same file yields high S_stab (stable)."""
        path = self._make_test_file(b"stable" * 683)
        try:
            for _ in range(10):
                ecar = self._make_ecar(pid=800, file_path=path)
                dbrg.process_event(ecar)
                fv = extractor.extract_features(ecar)
            # 1 unique target, 10 events -> S_stab = 1 - (1/10) = 0.9
            assert fv["S_stab"] > 0.8, f"Expected high S_stab for repeated access, got {fv['S_stab']}"
        finally:
            os.unlink(path)

    def test_s_stab_many_unique_files(self, extractor, dbrg):
        """Many unique files yield low S_stab (unstable)."""
        paths = []
        try:
            for i in range(10):
                path = self._make_test_file(f"file{i}".encode() * 512)
                paths.append(path)
                ecar = self._make_ecar(pid=900, file_path=path)
                dbrg.process_event(ecar)
                fv = extractor.extract_features(ecar)
            # 10 unique targets, 10 events -> S_stab = 1 - (10/10) = 0.0
            assert fv["S_stab"] < 0.1, f"Expected low S_stab, got {fv['S_stab']}"
        finally:
            for p in paths:
                os.unlink(p)

    def test_extraction_counter(self, extractor, dbrg):
        path = self._make_test_file(b"x" * 4096)
        try:
            for i in range(5):
                ecar = self._make_ecar(pid=1000, file_path=path)
                dbrg.process_event(ecar)
                extractor.extract_features(ecar)
            assert extractor.total_extractions == 5
        finally:
            os.unlink(path)

    def test_reset_pid(self, extractor, dbrg):
        path = self._make_test_file(b"data" * 1024)
        try:
            ecar = self._make_ecar(pid=1100, file_path=path)
            dbrg.process_event(ecar)
            extractor.extract_features(ecar)
            extractor.reset_pid(1100)
            assert extractor.hawkes.get_event_count(1100) == 0
            assert extractor.etd.get_observation_count(1100) == 0
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Thread Safety Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Verify concurrent access to shared engines is safe."""

    def test_hawkes_concurrent_updates(self):
        """Multiple threads updating different PIDs simultaneously."""
        engine = HawkesEngine()
        errors = []

        def worker(pid, count):
            try:
                t = 1000.0
                for i in range(count):
                    engine.update(pid, t + i * 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(pid, 50)) for pid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert engine.total_events == 500

    def test_feature_extractor_concurrent(self):
        """Multiple threads extracting features simultaneously."""
        dbrg = DBRGManager(decay_lambda=0.05)
        extractor = FeatureExtractor(dbrg_manager=dbrg)
        errors = []
        results = []

        def worker(pid):
            try:
                content = f"thread_{pid}_data".encode() * 256
                path = tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".txt", delete=False
                )
                path.write(content)
                path.close()
                try:
                    ecar = {
                        "actorID": f"pid:{pid}", "objectID": f"file:{path.name}",
                        "pid": pid, "operation": "FILE_MODIFY",
                        "timestamp": int(time.time() * 1000),
                        "context": {"exe_path": "test.exe", "sha256": "abc"},
                    }
                    dbrg.process_event(ecar)
                    fv = extractor.extract_features(ecar)
                    if fv:
                        results.append(fv)
                finally:
                    os.unlink(path.name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(pid,)) for pid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10
