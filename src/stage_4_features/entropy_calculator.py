"""
src/stage_4_features/entropy_calculator.py
============================================
Stage 4 — 4KB Buffer Sampling & Content Entropy Engine

Implements the core content-level signal extraction from the BIRF blueprint:

    Shannon Entropy:  H = -SUM P(x_i) * log2(P(x_i))   for i in 0..255

    Normalized Score: S_entropy = H / 8.0                 (maps to [0, 1])

Also computes 256-bin byte-frequency histogram moments (variance and kurtosis)
to distinguish encrypted random noise from structured text or compressed data.

Reference values:
    - All-zero / single-byte file  : H ~ 0.0,  S_entropy ~ 0.0
    - Structured English text      : H ~ 3.5-4.5, S_entropy ~ 0.45-0.55
    - Compressed data (ZIP/GZIP)   : H ~ 7.0-7.8, S_entropy ~ 0.88-0.98
    - Encrypted / random noise     : H ~ 7.99+,   S_entropy ~ 0.999+

This module is dependency-free (pure Python stdlib + math).
"""

import logging
import math
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Buffer size for 4KB post-write sampling (per blueprint)
BUFFER_SIZE: int = 4096


def read_file_head(file_path: str, size: int = BUFFER_SIZE) -> Optional[bytes]:
    """
    Non-blocking read of the first `size` bytes from a file.

    Parameters
    ----------
    file_path : str
        Absolute path to the target file.
    size : int
        Number of bytes to read (default 4096 per blueprint).

    Returns
    -------
    bytes or None
        The first `size` bytes of the file, or None if the file cannot be read.
    """
    if not file_path or file_path in ("UNKNOWN", "SYSTEM_PROCESS"):
        return None

    # Strip "file:" prefix if present (eCAR objectID format)
    clean_path = file_path
    if clean_path.startswith("file:"):
        clean_path = clean_path[5:]

    try:
        if not os.path.isfile(clean_path):
            return None
        with open(clean_path, "rb") as fh:
            data = fh.read(size)
        return data if len(data) > 0 else None
    except (PermissionError, OSError, IOError) as exc:
        logger.debug("[EntropyCalc] Cannot read %s: %s", clean_path, exc)
        return None


def shannon_entropy(data: bytes) -> float:
    """
    Compute Shannon entropy H over a byte buffer.

    Formula:  H = -SUM_{i=0}^{255} P(x_i) * log2(P(x_i))

    Parameters
    ----------
    data : bytes
        Raw byte buffer (typically 4KB from a file write).

    Returns
    -------
    float
        Entropy value in [0.0, 8.0] bits per byte.
        Returns 0.0 for empty input.
    """
    if not data:
        return 0.0

    length = len(data)

    # Build 256-bin byte frequency counts
    freq = [0] * 256
    for byte_val in data:
        freq[byte_val] += 1

    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)

    return entropy


def byte_histogram_stats(data: bytes) -> Tuple[float, float]:
    """
    Compute 256-bin byte-frequency histogram variance and kurtosis.

    These moments distinguish:
        - Plaintext:   Low variance, negative kurtosis (some bytes dominate)
        - Compressed:  High variance, near-zero kurtosis (fairly uniform)
        - Encrypted:   Very low variance, near-zero kurtosis (maximally uniform)

    Parameters
    ----------
    data : bytes
        Raw byte buffer.

    Returns
    -------
    (variance, kurtosis) : Tuple[float, float]
        variance  — Variance of the 256-bin normalized frequency distribution.
        kurtosis  — Excess kurtosis of the distribution (Fisher definition).
        Returns (0.0, 0.0) for empty input.
    """
    if not data:
        return 0.0, 0.0

    length = len(data)

    # Build 256-bin normalized frequency histogram
    freq = [0] * 256
    for byte_val in data:
        freq[byte_val] += 1

    probs = [count / length for count in freq]

    # Mean of the probability distribution
    mean = sum(probs) / 256.0  # = 1/256 for perfectly uniform

    # Variance = E[(p - mean)^2]
    variance = sum((p - mean) ** 2 for p in probs) / 256.0

    # Excess Kurtosis = E[(p - mean)^4] / variance^2 - 3
    if variance < 1e-15:
        # Near-zero variance = perfectly uniform or single-byte
        return variance, 0.0

    m4 = sum((p - mean) ** 4 for p in probs) / 256.0
    kurtosis = (m4 / (variance ** 2)) - 3.0

    return variance, kurtosis


def normalize_entropy(H: float) -> float:
    """
    Normalize raw Shannon entropy to S_entropy in [0.0, 1.0].

    Formula: S_entropy = H / 8.0

    Parameters
    ----------
    H : float
        Raw Shannon entropy in [0.0, 8.0].

    Returns
    -------
    float
        Normalized entropy score in [0.0, 1.0].
    """
    return max(0.0, min(1.0, H / 8.0))
