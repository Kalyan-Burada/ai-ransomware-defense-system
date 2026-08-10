"""
src/stage_4_features/__init__.py
==================================
Stage 4 — Multi-Layer Feature Extraction, 4KB Buffer Sampling
           & Hawkes Point Process

This package extracts per-event content-level and temporal-level threat
features from live file operations. It consumes enriched eCAR events
from Stage 1/2 and queries the Stage 3 DBRG graph for structural metrics.

Sub-Modules
-----------
entropy_calculator  — 4KB buffer sampling, Shannon entropy, histogram stats.
hawkes_engine       — Hawkes self-exciting point process for burstiness.
etd_engine          — Entropy-Topology Divergence (2D KL-divergence).
feature_extractor   — Orchestrator producing complete feature vectors.

Output Feature Vector:
    S_entropy, S_ETD, S_dist, S_dev, S_stab, lambda_norm, branching_n

Exports
-------
FeatureExtractor   — Main orchestrator class.
HawkesEngine       — Per-process Hawkes intensity tracker.
ETDEngine          — ETD KL-divergence calculator.
shannon_entropy    — Pure-math Shannon entropy function.
normalize_entropy  — Maps H to [0, 1].
read_file_head     — Non-blocking 4KB file buffer reader.
"""

__version__ = "1.0.0"
__stage__ = "Stage 4: Multi-Layer Feature Extraction & Hawkes Point Process"

from src.stage_4_features.entropy_calculator import (
    shannon_entropy,
    normalize_entropy,
    byte_histogram_stats,
    read_file_head,
    BUFFER_SIZE,
)
from src.stage_4_features.hawkes_engine import HawkesEngine
from src.stage_4_features.etd_engine import ETDEngine
from src.stage_4_features.feature_extractor import FeatureExtractor

__all__ = [
    "FeatureExtractor",
    "HawkesEngine",
    "ETDEngine",
    "shannon_entropy",
    "normalize_entropy",
    "byte_histogram_stats",
    "read_file_head",
    "BUFFER_SIZE",
]
