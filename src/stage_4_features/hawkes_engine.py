"""
src/stage_4_features/hawkes_engine.py
=======================================
Stage 4 — Hawkes Self-Exciting Point Process Engine

Implements the temporal burstiness detection model from the BIRF blueprint:

    Intensity:    lambda(t) = mu + SUM_{t_i < t} alpha * exp(-beta * (t - t_i))

    Branching Ratio:  n = alpha / beta
        - n < 1.0 : sub-critical (events decay naturally — benign)
        - n >= 1.0: super-critical / "superheated" (self-exciting — attack state)

    Normalized Intensity:
        lambda_norm(t) = min(1.0, (lambda(t) - mu) / lambda_max)

The engine maintains per-process event timestamp histories with automatic
pruning of old timestamps beyond a configurable time horizon to bound memory.

Default Parameters (from blueprint calibration notes):
    mu    = 0.1    (background / baseline event rate)
    alpha = 0.5    (excitation magnitude per event)
    beta  = 0.8    (exponential decay rate of excitation)
    -> branching ratio n = 0.5/0.8 = 0.625 < 1.0 (sub-critical baseline)

For ransomware-like burst patterns, effective n can exceed 1.0 when
events arrive faster than beta can decay prior excitations.
"""

import logging
import math
import time
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HawkesEngine:
    """
    Per-process Hawkes self-exciting point process tracker.

    Parameters
    ----------
    mu : float
        Background (baseline) event intensity rate.
    alpha : float
        Excitation magnitude — how much each event boosts intensity.
    beta : float
        Excitation decay rate — how fast the boost fades.
    lambda_max : float
        Saturation cap for normalization (default 10.0).
    horizon_sec : float
        Maximum lookback window for event timestamps (default 120.0s).
        Events older than this are pruned to bound memory.
    """

    def __init__(
        self,
        mu: float = 0.1,
        alpha: float = 0.5,
        beta: float = 0.8,
        lambda_max: float = 10.0,
        horizon_sec: float = 120.0,
    ) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.mu: float = mu
        self.alpha: float = alpha
        self.beta: float = beta
        self.lambda_max: float = lambda_max
        self.horizon_sec: float = horizon_sec

        # Per-process state: pid -> list of event timestamps
        self._events: Dict[int, List[float]] = {}
        self._lock = threading.Lock()

        # Statistics
        self.total_events: int = 0

    @property
    def branching_ratio(self) -> float:
        """
        Return the branching ratio n = alpha / beta.

        n < 1.0 means sub-critical (benign — events naturally decay).
        n >= 1.0 means super-critical (self-exciting attack state).
        """
        return self.alpha / self.beta

    def update(self, pid: int, t_now: Optional[float] = None) -> float:
        """
        Register a new event for process `pid` at time `t_now`.

        Parameters
        ----------
        pid : int
            Process ID.
        t_now : float, optional
            Event timestamp (Unix seconds). Defaults to time.time().

        Returns
        -------
        float
            The updated Hawkes intensity lambda(t) for this process.
        """
        if t_now is None:
            t_now = time.time()

        with self._lock:
            if pid not in self._events:
                self._events[pid] = []

            self._events[pid].append(t_now)
            self.total_events += 1

            # Prune old events beyond the horizon
            cutoff = t_now - self.horizon_sec
            self._events[pid] = [
                t for t in self._events[pid] if t > cutoff
            ]

        return self.get_intensity(pid, t_now)

    def get_intensity(self, pid: int, t_now: Optional[float] = None) -> float:
        """
        Compute the current Hawkes intensity for process `pid`.

        Formula: lambda(t) = mu + SUM_{t_i < t} alpha * exp(-beta * (t - t_i))

        Parameters
        ----------
        pid : int
            Process ID.
        t_now : float, optional
            Current timestamp. Defaults to time.time().

        Returns
        -------
        float
            Current intensity lambda(t). Returns mu if no events exist.
        """
        if t_now is None:
            t_now = time.time()

        with self._lock:
            events = self._events.get(pid, [])
            if not events:
                return self.mu

            excitation = 0.0
            for t_i in events:
                dt = t_now - t_i
                if dt > 0:
                    excitation += self.alpha * math.exp(-self.beta * dt)

            return self.mu + excitation

    def get_normalized_intensity(self, pid: int, t_now: Optional[float] = None) -> float:
        """
        Return the normalized intensity in [0.0, 1.0].

        Formula: lambda_norm(t) = min(1.0, (lambda(t) - mu) / lambda_max)

        Parameters
        ----------
        pid : int
            Process ID.
        t_now : float, optional
            Current timestamp. Defaults to time.time().

        Returns
        -------
        float
            Normalized intensity in [0.0, 1.0].
        """
        raw = self.get_intensity(pid, t_now)
        if self.lambda_max <= 0:
            return 0.0
        normalized = (raw - self.mu) / self.lambda_max
        return max(0.0, min(1.0, normalized))

    def is_superheated(self) -> bool:
        """
        Return True if the branching ratio n >= 1.0 (super-critical state).

        In super-critical state, each event generates on average >= 1 child
        event, causing self-exciting runaway intensity — characteristic of
        ransomware encryption bursts.
        """
        return self.branching_ratio >= 1.0

    def get_event_count(self, pid: int) -> int:
        """Return the number of tracked events for a given PID."""
        with self._lock:
            return len(self._events.get(pid, []))

    def get_all_pids(self) -> List[int]:
        """Return all PIDs currently being tracked."""
        with self._lock:
            return list(self._events.keys())

    def reset_pid(self, pid: int) -> None:
        """Clear all event history for a specific PID."""
        with self._lock:
            self._events.pop(pid, None)

    def __repr__(self) -> str:
        return (
            f"HawkesEngine(mu={self.mu}, alpha={self.alpha}, beta={self.beta}, "
            f"n={self.branching_ratio:.3f}, pids_tracked={len(self._events)}, "
            f"total_events={self.total_events})"
        )
