"""
src/stage_4_features/etd_engine.py
====================================
Stage 4 — Entropy-Topology Divergence (ETD) Engine

Implements the joint entropy-topology divergence metric from the BIRF blueprint:

    S_ETD = D_KL( P(delta_H, d_out | t) || P_benign )

Where:
    delta_H  = H_post - H_pre   (entropy change from the write operation)
    d_out    = process fan-out degree from DBRG (# unique files accessed)
    P_benign = pre-learned baseline joint distribution from dev-tool traffic

The engine maintains a rolling 2D joint histogram per process of
(delta_H_bin, d_out_bin) observations and computes KL-divergence against
the benign baseline to detect divergent behavior patterns.

Benign Baseline Design:
    Built from representative developer-tool and archiver traffic patterns:
    - Text editors (Notepad, VS Code): Low delta_H, low d_out
    - Compilers (gcc, cl.exe): Moderate delta_H, moderate d_out
    - Archivers (7-Zip, WinRAR): High delta_H, low d_out
    - Browsers (Chrome, Edge): Low delta_H, low d_out

    Ransomware Pattern (detected as high S_ETD):
    - Encryption: Very high delta_H (plaintext -> ciphertext), very high d_out
"""

import logging
import math
import threading
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ─── Binning Constants ────────────────────────────────────────────────────────

# delta_H bins: discretize entropy change into N_DH bins over [-8.0, +8.0]
N_DH_BINS: int = 16
DH_MIN: float = -8.0
DH_MAX: float = 8.0

# d_out bins: discretize fan-out degree into N_DOUT bins over [0, MAX_DOUT]
N_DOUT_BINS: int = 10
MAX_DOUT: int = 50  # Cap at 50; anything above maps to the last bin


def _bin_delta_h(delta_h: float) -> int:
    """Map delta_H value to a histogram bin index."""
    clamped = max(DH_MIN, min(DH_MAX, delta_h))
    bin_width = (DH_MAX - DH_MIN) / N_DH_BINS
    idx = int((clamped - DH_MIN) / bin_width)
    return min(idx, N_DH_BINS - 1)


def _bin_d_out(d_out: int) -> int:
    """Map fan-out degree to a histogram bin index."""
    clamped = max(0, min(MAX_DOUT, d_out))
    bin_width = (MAX_DOUT + 1) / N_DOUT_BINS
    idx = int(clamped / bin_width)
    return min(idx, N_DOUT_BINS - 1)


def _build_benign_baseline() -> List[List[float]]:
    """
    Build the pre-trained benign baseline 2D distribution P_benign.

    This models typical developer/office workstation traffic:
    - Most activity: low delta_H (0-2 range), low d_out (1-5 files)
    - Some activity: moderate delta_H from compilers
    - Rare activity: high delta_H from archivers (but still low d_out)
    - Almost zero  : high delta_H + high d_out (ransomware signature)

    Returns a 2D probability distribution [N_DH_BINS][N_DOUT_BINS].
    """
    raw = [[0.0] * N_DOUT_BINS for _ in range(N_DH_BINS)]

    # Center bins (delta_H ~ 0): Most benign activity has small entropy changes
    # and low fan-out. Weight these heavily.
    for dh_bin in range(N_DH_BINS // 2 - 2, N_DH_BINS // 2 + 3):
        for dout_bin in range(min(4, N_DOUT_BINS)):
            # Higher weight for small delta_H + small d_out
            dh_distance = abs(dh_bin - N_DH_BINS // 2)
            dout_distance = dout_bin
            raw[dh_bin][dout_bin] = max(0.01, 10.0 / (1.0 + dh_distance + dout_distance * 2))

    # Archiver pattern: high delta_H but low d_out (7-Zip creates 1 archive)
    for dh_bin in range(N_DH_BINS - 4, N_DH_BINS):
        raw[dh_bin][0] = 1.5
        raw[dh_bin][1] = 0.5

    # Compiler pattern: moderate delta_H, moderate d_out (several .obj files)
    for dh_bin in range(N_DH_BINS // 2 + 1, N_DH_BINS // 2 + 4):
        for dout_bin in range(2, min(5, N_DOUT_BINS)):
            raw[dh_bin][dout_bin] = 1.0

    # Near-zero probability for high delta_H + high d_out (ransomware zone)
    for dh_bin in range(N_DH_BINS - 4, N_DH_BINS):
        for dout_bin in range(N_DOUT_BINS // 2, N_DOUT_BINS):
            raw[dh_bin][dout_bin] = 0.001

    # Normalize to a valid probability distribution
    total = sum(sum(row) for row in raw)
    if total > 0:
        for i in range(N_DH_BINS):
            for j in range(N_DOUT_BINS):
                raw[i][j] /= total

    # Apply Laplace smoothing to avoid log(0) in KL-divergence
    epsilon = 1e-8
    for i in range(N_DH_BINS):
        for j in range(N_DOUT_BINS):
            raw[i][j] = max(epsilon, raw[i][j])

    # Re-normalize after smoothing
    total = sum(sum(row) for row in raw)
    for i in range(N_DH_BINS):
        for j in range(N_DOUT_BINS):
            raw[i][j] /= total

    return raw


def _kl_divergence_2d(
    p: List[List[float]], q: List[List[float]]
) -> float:
    """
    Compute KL-divergence D_KL(P || Q) between two 2D distributions.

    D_KL(P || Q) = SUM_i SUM_j P(i,j) * log2( P(i,j) / Q(i,j) )

    Both P and Q must have the same dimensions [N_DH_BINS][N_DOUT_BINS].
    """
    kl = 0.0
    for i in range(len(p)):
        for j in range(len(p[0])):
            p_val = p[i][j]
            q_val = q[i][j]
            if p_val > 1e-12 and q_val > 1e-12:
                kl += p_val * math.log2(p_val / q_val)
    return max(0.0, kl)


class ETDEngine:
    """
    Entropy-Topology Divergence engine.

    Maintains a rolling 2D joint histogram per process of (delta_H, d_out)
    observations and computes KL-divergence against the benign baseline.

    Parameters
    ----------
    window_size : int
        Maximum number of recent observations per process (default 50).
    s_etd_cap : float
        Maximum S_ETD score cap for normalization (default 10.0).
    """

    def __init__(self, window_size: int = 50, s_etd_cap: float = 10.0) -> None:
        self.window_size: int = window_size
        self.s_etd_cap: float = s_etd_cap
        self._baseline: List[List[float]] = _build_benign_baseline()

        # Per-process rolling observation windows: pid -> list of (delta_h, d_out)
        self._observations: Dict[int, List[Tuple[float, int]]] = {}
        # Per-process previous entropy for delta_H calculation
        self._prev_entropy: Dict[int, float] = {}
        self._lock = threading.Lock()

    def get_previous_entropy(self, pid: int) -> float:
        """Return the last observed entropy for a process (for delta_H calc)."""
        with self._lock:
            return self._prev_entropy.get(pid, 0.0)

    def set_previous_entropy(self, pid: int, entropy: float) -> None:
        """Store the current entropy as the previous for next delta_H."""
        with self._lock:
            self._prev_entropy[pid] = entropy

    def compute_etd(self, pid: int, delta_h: float, d_out: int) -> float:
        """
        Record a new (delta_H, d_out) observation and compute S_ETD.

        Parameters
        ----------
        pid : int
            Process ID.
        delta_h : float
            Entropy change H_post - H_pre for this write event.
        d_out : int
            Process fan-out degree from the DBRG graph.

        Returns
        -------
        float
            Normalized S_ETD score in [0.0, 1.0].
        """
        with self._lock:
            if pid not in self._observations:
                self._observations[pid] = []

            self._observations[pid].append((delta_h, d_out))

            # Trim to rolling window
            if len(self._observations[pid]) > self.window_size:
                self._observations[pid] = self._observations[pid][-self.window_size:]

            # Build the live 2D histogram from recent observations
            live_hist = [[0.0] * N_DOUT_BINS for _ in range(N_DH_BINS)]
            for dh, do in self._observations[pid]:
                dh_bin = _bin_delta_h(dh)
                do_bin = _bin_d_out(do)
                live_hist[dh_bin][do_bin] += 1.0

            # Normalize to probability distribution with Laplace smoothing
            total = sum(sum(row) for row in live_hist)
            epsilon = 1e-8
            if total > 0:
                for i in range(N_DH_BINS):
                    for j in range(N_DOUT_BINS):
                        live_hist[i][j] = max(epsilon, live_hist[i][j] / total)
            else:
                # No observations yet — use uniform
                uniform = 1.0 / (N_DH_BINS * N_DOUT_BINS)
                for i in range(N_DH_BINS):
                    for j in range(N_DOUT_BINS):
                        live_hist[i][j] = uniform

            # Re-normalize after smoothing
            total = sum(sum(row) for row in live_hist)
            for i in range(N_DH_BINS):
                for j in range(N_DOUT_BINS):
                    live_hist[i][j] /= total

        # Compute KL-divergence: D_KL( P_live || P_benign )
        raw_etd = _kl_divergence_2d(live_hist, self._baseline)

        # Normalize to [0, 1] using cap
        s_etd = min(1.0, raw_etd / self.s_etd_cap)

        logger.debug(
            "[ETD] PID=%d  delta_H=%.4f  d_out=%d  raw_KL=%.6f  S_ETD=%.4f",
            pid, delta_h, d_out, raw_etd, s_etd,
        )

        return s_etd

    def get_observation_count(self, pid: int) -> int:
        """Return the number of tracked observations for a PID."""
        with self._lock:
            return len(self._observations.get(pid, []))

    def reset_pid(self, pid: int) -> None:
        """Clear all state for a specific PID."""
        with self._lock:
            self._observations.pop(pid, None)
            self._prev_entropy.pop(pid, None)

    def __repr__(self) -> str:
        return (
            f"ETDEngine(window={self.window_size}, "
            f"pids_tracked={len(self._observations)})"
        )
