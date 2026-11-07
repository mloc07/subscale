"""Evaluation metric suite for AI Dubbing module.

Metrics:
1. Voice Similarity: Cosine similarity of ECAPA-TDNN speaker embeddings between reference and generated speech.
2. Word Error Rate (WER) / Intelligibility: Inverse ASR transcription accuracy vs expected translation text.
3. Duration Drift: Absolute and relative timing deviation between generated audio duration and target duration budget.
4. Diarization Error Rate (DER) / Speaker Confusion: Misassigned speaker segments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def compute_duration_drift(actual_sec: float, target_sec: float) -> dict[str, float]:
    """Compute duration drift metrics."""
    diff = actual_sec - target_sec
    ratio = actual_sec / target_sec if target_sec > 0 else 1.0
    return {
        "actual_sec": round(actual_sec, 3),
        "target_sec": round(target_sec, 3),
        "abs_diff_sec": round(abs(diff), 3),
        "drift_ratio": round(ratio, 3),
        "is_acceptable": abs(diff) <= 0.6 or (0.85 <= ratio <= 1.25)
    }


def compute_simple_wer(hypothesis: str, reference: str) -> float:
    """Compute word error rate (Levenshtein distance at word level)."""
    h_words = hypothesis.strip().lower().split()
    r_words = reference.strip().lower().split()
    
    if not r_words:
        return 1.0 if h_words else 0.0
        
    d = np.zeros((len(r_words) + 1, len(h_words) + 1), dtype=np.int32)
    for i in range(len(r_words) + 1):
        d[i, 0] = i
    for j in range(len(h_words) + 1):
        d[0, j] = j
        
    for i in range(1, len(r_words) + 1):
        for j in range(1, len(h_words) + 1):
            if r_words[i - 1] == h_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
                
    return float(d[len(r_words), len(h_words)] / len(r_words))


def evaluate_mock_pipeline() -> dict:
    """Run evaluation on mock inputs for CI/testing without heavy models."""
    # Synthetic 192-dim embeddings
    np.random.seed(42)
    emb_ref = np.random.randn(192)
    # Similar voice has small noise
    emb_gen_similar = emb_ref + np.random.randn(192) * 0.3
    # Different voice
    emb_gen_diff = np.random.randn(192)
    
    sim_match = compute_cosine_similarity(emb_ref, emb_gen_similar)
    sim_diff = compute_cosine_similarity(emb_ref, emb_gen_diff)
    
    drift = compute_duration_drift(actual_sec=3.9, target_sec=4.0)
    wer = compute_simple_wer("làm sao biết mình có gì nếu không thử", "làm sao biết được mình có gì nếu không thử sức")
    
    return {
        "status": "ok",
        "mock_voice_similarity_same": round(sim_match, 4),
        "mock_voice_similarity_different": round(sim_diff, 4),
        "mock_duration_drift": drift,
        "mock_wer": round(wer, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="AI Dubbing Evaluation Suite")
    parser.add_argument("--mock", action="store_true", help="Run with mock data (for CI tests)")
    args = parser.parse_args()
    
    if args.mock:
        result = evaluate_mock_pipeline()
        print("Mock Evaluation Results:\n", json.dumps(result, indent=2))
    else:
        print("Run with --mock or pass audio files for evaluation.")


if __name__ == "__main__":
    main()
