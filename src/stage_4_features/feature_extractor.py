"""
src/stage_4_features/feature_extractor.py
===========================================
Stage 4 — Multi-Layer Feature Extraction Orchestrator

This is the main entry point for Stage 4 that ties all sub-modules together.
It consumes enriched eCAR events from Stage 1/2 and queries the Stage 3
DBRG graph to produce a complete per-event feature vector.

Per-event output (FeatureVector):
    S_entropy     : Normalized Shannon entropy of the 4KB post-write buffer [0, 1]
    S_ETD         : Entropy-Topology Divergence score [0, 1]
    S_dist        : Graph Fan-Out Distance (normalized process out-degree) [0, 1]
    S_dev         : Trajectory Velocity (rate of new file targets / second) [0, 1]
    S_stab        : Operational Stability (ratio of repeated vs new interactions) [0, 1]
    lambda_norm   : Normalized Hawkes intensity [0, 1]
    branching_n   : Hawkes branching ratio (alpha/beta)
    is_superheated: True if branching_n >= 1.0
    entropy_raw   : Raw Shannon entropy H [0, 8]
    hist_variance : 256-bin byte histogram variance
    hist_kurtosis : 256-bin byte histogram kurtosis

Data Flow:
    Stage 1 (eCAR event) + Stage 3 (DBRG graph)
        -> Stage 4 FeatureExtractor.extract_features()
            -> FeatureVector dict
                -> feeds Stage 5 (Anomaly Scoring) & Stage 6 (Threat Fusion)
"""

import logging
import os
import time
import threading
from typing import Any, Dict, Optional

from src.stage_4_features.entropy_calculator import (
    read_file_head,
    shannon_entropy,
    byte_histogram_stats,
    normalize_entropy,
    BUFFER_SIZE,
)
from src.stage_4_features.hawkes_engine import HawkesEngine
from src.stage_4_features.etd_engine import ETDEngine

logger = logging.getLogger(__name__)

# Maximum fan-out degree for normalization
MAX_FANOUT: int = 50

# Maximum new-target rate (files/sec) for S_dev normalization
MAX_TARGET_RATE: float = 20.0


class FeatureExtractor:
    """
    Stage 4 orchestrator: extracts multi-layer features from live eCAR events.

    Parameters
    ----------
    dbrg_manager : DBRGManager
        Reference to the Stage 3 DBRG graph manager for live graph queries.
    hawkes_mu : float
        Hawkes background rate (default 0.1).
    hawkes_alpha : float
        Hawkes excitation magnitude (default 0.5).
    hawkes_beta : float
        Hawkes excitation decay rate (default 0.8).
    etd_window : int
        Rolling window size for ETD observations (default 50).
    """

    def __init__(
        self,
        dbrg_manager=None,
        hawkes_mu: float = 0.1,
        hawkes_alpha: float = 0.5,
        hawkes_beta: float = 0.8,
        etd_window: int = 50,
    ) -> None:
        self.dbrg = dbrg_manager
        self.hawkes = HawkesEngine(
            mu=hawkes_mu, alpha=hawkes_alpha, beta=hawkes_beta,
        )
        self.etd = ETDEngine(window_size=etd_window)

        # Per-process trajectory tracking for S_dev and S_stab
        # pid -> {"first_seen": float, "unique_targets": set, "total_events": int}
        self._trajectory: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Statistics
        self.total_extractions: int = 0
        self.total_files_sampled: int = 0

    def extract_features(self, ecar_event: dict) -> Optional[Dict[str, Any]]:
        """
        Extract the full multi-layer feature vector from one eCAR event.

        Parameters
        ----------
        ecar_event : dict
            A Stage 1/2 enriched eCAR event dict with keys:
            actorID, objectID, pid, operation, context, timestamp.

        Returns
        -------
        dict or None
            The complete FeatureVector dict, or None if extraction fails.
        """
        try:
            # ── 1. Parse event metadata ──────────────────────────────────
            pid: int = ecar_event.get("pid", -1)
            operation: str = ecar_event.get("operation", "")
            object_id: str = ecar_event.get("objectID", "")
            actor_id: str = ecar_event.get("actorID", f"pid:{pid}")
            timestamp: float = ecar_event.get("timestamp", time.time() * 1000)
            context: dict = ecar_event.get("context", {})

            # Extract file path from objectID
            file_path = object_id
            if file_path.startswith("file:"):
                file_path = file_path[5:]

            t_now = timestamp / 1000.0 if timestamp > 1e12 else timestamp

            # ── 2. 4KB Buffer Sampling & Content Entropy ─────────────────
            data = read_file_head(file_path, BUFFER_SIZE)

            if data is not None:
                entropy_raw = shannon_entropy(data)
                hist_variance, hist_kurtosis = byte_histogram_stats(data)
                self.total_files_sampled += 1
            else:
                # File not readable (deleted, permission, etc.)
                entropy_raw = 0.0
                hist_variance = 0.0
                hist_kurtosis = 0.0

            s_entropy = normalize_entropy(entropy_raw)

            # ── 3. Query DBRG for graph topology metrics ─────────────────
            d_out = 0
            if self.dbrg is not None:
                d_out = self._get_process_fanout(actor_id)

            # ── 4. Compute delta_H and ETD ───────────────────────────────
            prev_h = self.etd.get_previous_entropy(pid)
            delta_h = entropy_raw - prev_h
            self.etd.set_previous_entropy(pid, entropy_raw)

            s_etd = self.etd.compute_etd(pid, delta_h, d_out)

            # ── 5. Update Hawkes point process ───────────────────────────
            self.hawkes.update(pid, t_now)
            lambda_norm = self.hawkes.get_normalized_intensity(pid, t_now)
            branching_n = self.hawkes.branching_ratio
            superheated = self.hawkes.is_superheated()

            # ── 6. Compute graph trajectory metrics ──────────────────────
            s_dist = self._compute_s_dist(d_out)
            s_dev = self._compute_s_dev(pid, file_path, t_now)
            s_stab = self._compute_s_stab(pid, file_path)

            self.total_extractions += 1

            feature_vector = {
                # ── Primary normalized scores (feed Stage 5/6) ───────────
                "S_entropy": s_entropy,
                "S_ETD": s_etd,
                "S_dist": s_dist,
                "S_dev": s_dev,
                "S_stab": s_stab,

                # ── Hawkes temporal signals ───────────────────────────────
                "lambda_norm": lambda_norm,
                "branching_n": branching_n,
                "is_superheated": superheated,

                # ── Raw diagnostic values ────────────────────────────────
                "entropy_raw": entropy_raw,
                "delta_H": delta_h,
                "hist_variance": hist_variance,
                "hist_kurtosis": hist_kurtosis,
                "fan_out_degree": d_out,

                # ── Event context (passthrough) ──────────────────────────
                "pid": pid,
                "file_path": file_path,
                "operation": operation,
                "timestamp": timestamp,
                "actor_id": actor_id,
            }

            logger.debug(
                "[Stage4] PID=%d  S_ent=%.3f  S_ETD=%.3f  S_dist=%.3f  "
                "S_dev=%.3f  S_stab=%.3f  lambda=%.3f  n=%.3f  %s",
                pid, s_entropy, s_etd, s_dist, s_dev, s_stab,
                lambda_norm, branching_n,
                "SUPERHEATED!" if superheated else "normal",
            )

            return feature_vector

        except Exception as exc:
            logger.error(
                "[Stage4] Feature extraction failed: %s", exc, exc_info=True
            )
            return None

    # ── Graph Topology Helpers ────────────────────────────────────────────

    def _get_process_fanout(self, actor_id: str) -> int:
        """
        Query the DBRG for the out-degree (fan-out) of a process node.

        This counts the number of distinct file nodes that the process
        currently has active edges to in the graph.
        """
        try:
            with self.dbrg.lock:
                if self.dbrg.graph.has_node(actor_id):
                    return self.dbrg.graph.out_degree(actor_id)
            return 0
        except Exception:
            return 0

    def _compute_s_dist(self, d_out: int) -> float:
        """
        Compute S_dist (Graph Fan-Out Distance).

        Normalized process out-degree: S_dist = min(1.0, d_out / MAX_FANOUT)

        Benign apps typically have d_out in [1, 5].
        Ransomware drives d_out toward [10, 1000+].
        """
        return min(1.0, d_out / MAX_FANOUT)

    def _compute_s_dev(self, pid: int, file_path: str, t_now: float) -> float:
        """
        Compute S_dev (Trajectory Velocity).

        Measures the rate at which the process is accessing NEW unique file
        targets per second. High S_dev = rapidly expanding attack surface.

        S_dev = min(1.0, unique_targets / elapsed_time / MAX_TARGET_RATE)
        """
        with self._lock:
            if pid not in self._trajectory:
                self._trajectory[pid] = {
                    "first_seen": t_now,
                    "unique_targets": set(),
                    "total_events": 0,
                }

            traj = self._trajectory[pid]
            traj["unique_targets"].add(file_path.lower())
            traj["total_events"] += 1

            elapsed = max(0.1, t_now - traj["first_seen"])
            unique_count = len(traj["unique_targets"])

            rate = unique_count / elapsed
            return min(1.0, rate / MAX_TARGET_RATE)

    def _compute_s_stab(self, pid: int, file_path: str) -> float:
        """
        Compute S_stab (Operational Stability).

        Measures the ratio of repeated interactions vs unique targets.
        High S_stab (near 1.0) = stable benign behavior (same files repeatedly).
        Low S_stab (near 0.0) = unstable (many new files = suspicious).

        S_stab = 1.0 - (unique_targets / total_events)
        """
        with self._lock:
            traj = self._trajectory.get(pid)
            if not traj or traj["total_events"] == 0:
                return 0.5  # Neutral for first observation

            unique = len(traj["unique_targets"])
            total = traj["total_events"]

            # Ratio of unique targets to total events
            uniqueness_ratio = unique / total

            # Invert: high uniqueness = low stability
            return max(0.0, min(1.0, 1.0 - uniqueness_ratio))

    def reset_pid(self, pid: int) -> None:
        """Clear all tracked state for a specific PID."""
        self.hawkes.reset_pid(pid)
        self.etd.reset_pid(pid)
        with self._lock:
            self._trajectory.pop(pid, None)

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor(extractions={self.total_extractions}, "
            f"files_sampled={self.total_files_sampled}, "
            f"hawkes={self.hawkes}, etd={self.etd})"
        )
