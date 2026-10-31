"""Evaluation script for ASR (Whisper) measuring WER and CER against reference text.

Supports normalization (NFKC, lowercase, punctuation removal) and benchmarking
different ASR configurations.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Ensure repo root and src are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import jiwer
except ImportError:
    jiwer = None

from ecoup.subtitle.asr import AsrResult, FasterWhisperBackend
from ecoup.subtitle.audio import extract_audio
from ecoup.subtitle.config import AsrConfig


def normalize_text_for_eval(text: str) -> str:
    """Normalize text for speech recognition evaluation.

    - Unicode NFKC normalization
    - Lowercase
    - Remove punctuation (keep apostrophes if needed, or remove uniformly)
    - Collapse multiple spaces into single space
    """
    if not text:
        return ""
    # NFKC
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower()
    # Replace newlines and tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove punctuation
    # Remove common typographic quotes/symbols
    text = re.sub(r"[’‘'\"“”«»„…—–\-.,!?;:()[\]{}<>/*+=\\_~`^|#$@%&]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute WER and CER using jiwer."""
    norm_hyp = normalize_text_for_eval(hypothesis)
    norm_ref = normalize_text_for_eval(reference)

    if not norm_ref:
        return {"wer": 1.0 if norm_hyp else 0.0, "cer": 1.0 if norm_hyp else 0.0}

    if jiwer is None:
        # Simple fallback token error rate if jiwer is missing
        ref_words = norm_ref.split()
        hyp_words = norm_hyp.split()
        import difflib
        sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
        wer = 1.0 - sm.ratio()
        sm_char = difflib.SequenceMatcher(None, norm_ref, norm_hyp)
        cer = 1.0 - sm_char.ratio()
        return {"wer": round(wer, 4), "cer": round(cer, 4)}

    wer = float(jiwer.wer(norm_ref, norm_hyp))
    cer = float(jiwer.cer(norm_ref, norm_hyp))
    return {"wer": round(wer, 4), "cer": round(cer, 4)}


def run_single_eval(
    media_path: str | Path,
    reference_path: str | Path,
    asr_cfg: AsrConfig,
    work_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Extract audio, run ASR with given AsrConfig, and compute WER/CER against reference."""
    media_path = Path(media_path)
    ref_text = Path(reference_path).read_text(encoding="utf-8")

    if work_dir is None:
        work_dir = Path("eval/asr/temp")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract audio if needed
    if media_path.suffix.lower() == ".wav":
        wav_path = media_path
    else:
        wav_path = work_dir / f"{media_path.stem}_audio.wav"
        if not wav_path.exists():
            print(f"Extracting 16kHz mono audio from {media_path.name}...")
            extract_audio(media_path, wav_path)

    # 2. Run ASR
    backend = FasterWhisperBackend()
    print(f"Running ASR with model={asr_cfg.model}, audio_mode={asr_cfg.audio_mode}, "
          f"multilingual_chunking={asr_cfg.multilingual_chunking}, lang={asr_cfg.language}...")
    t0 = time.time()
    res: AsrResult = backend.transcribe(str(wav_path), asr_cfg)
    elapsed = time.time() - t0

    # 3. Assemble hypothesis text
    hyp_text = " ".join(s.text.strip() for s in res.segments if s.text.strip())

    # 4. Metrics
    metrics = compute_metrics(hyp_text, ref_text)

    # 5. Detected languages summary
    seg_langs = [s.language for s in res.segments if s.language]
    from collections import Counter
    lang_counts = dict(Counter(seg_langs))

    return {
        "media": str(media_path),
        "duration_s": round(res.duration_s, 2),
        "elapsed_s": round(elapsed, 2),
        "peak_vram_gb": res.peak_vram_gb,
        "segment_count": len(res.segments),
        "detected_language": res.language,
        "language_probability": round(res.language_probability, 3),
        "language_distribution": lang_counts,
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "hyp_text": hyp_text,
        "segments": [asdict(s) for s in res.segments],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper ASR accuracy (WER/CER)")
    parser.add_argument("--media", required=True, help="Path to input video or audio file")
    parser.add_argument("--ref", required=True, help="Path to reference text file")
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument("--audio-mode", default="speech", choices=["speech", "music", "off"], help="Audio VAD mode")
    parser.add_argument("--language", default=None, help="Force language (e.g. en, vi) or None for auto")
    parser.add_argument("--chunking", action="store_true", default=False, help="Enable legacy 25s multilingual chunking")
    parser.add_argument("--condition-on-prev", action="store_true", default=False, help="Condition on previous text")
    parser.add_argument("--initial-prompt", default=None, help="Initial prompt / vocabulary hint")
    parser.add_argument("--out", default=None, help="JSON output file to save results")
    args = parser.parse_args()

    cfg = AsrConfig(
        model=args.model,
        audio_mode=args.audio_mode,
        language=args.language,
        multilingual_chunking=args.chunking,
        condition_on_previous_text=args.condition_on_prev,
        initial_prompt=args.initial_prompt,
    )

    result = run_single_eval(args.media, args.ref, cfg)

    print("\n" + "=" * 60)
    print("ASR EVALUATION RESULTS:")
    print(f"Media Duration   : {result['duration_s']}s")
    print(f"Elapsed Time     : {result['elapsed_s']}s")
    print(f"Peak VRAM        : {result['peak_vram_gb']} GB")
    print(f"Segment Count    : {result['segment_count']}")
    print(f"Overall Language : {result['detected_language']} (p={result['language_probability']})")
    print(f"Lang Distribution: {result['language_distribution']}")
    print(f"WER (Word Error) : {result['wer'] * 100:.2f}%")
    print(f"CER (Char Error) : {result['cer'] * 100:.2f}%")
    print("=" * 60)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved detailed result to {out_path}")


if __name__ == "__main__":
    main()
