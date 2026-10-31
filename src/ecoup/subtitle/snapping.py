"""Lyric snapping: align reference lyrics to ASR timestamps via fuzzy sequence matching."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Optional

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    import difflib

    class _FuzzFallback:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100.0

        @staticmethod
        def token_set_ratio(s1: str, s2: str) -> float:
            w1 = " ".join(sorted(set(s1.split())))
            w2 = " ".join(sorted(set(s2.split())))
            return difflib.SequenceMatcher(None, w1, w2).ratio() * 100.0

    fuzz = _FuzzFallback()  # type: ignore

from .segments import Segment, clean_text


def _normalize_lyric_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).lower()
    line = re.sub(r"[’‘'\"“”«»„…—–\-.,!?;:()[\]{}<>/*+=\\_~`^|#$@%&]", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def parse_raw_lyrics(lyrics_text: str) -> list[str]:
    """Split lyrics into meaningful lines, stripping header tags like [Verse], [Chorus] etc."""
    raw_lines = lyrics_text.splitlines()
    clean_lines: list[str] = []
    tag_pattern = re.compile(
        r"^\s*(\[|\()? *(verse|chorus|pre-chorus|bridge|outro|intro|hook|đoạn|điệp khúc|lời \d+)[\s\d:.-]*(\]|\))?\s*$",
        re.IGNORECASE,
    )
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        if tag_pattern.match(line):
            continue
        # Remove parenthetical tags at line start/end if purely metadata
        clean_lines.append(line)
    return clean_lines


def snap_lyrics_to_segments(
    segments: list[Segment],
    lyrics_text: str,
    threshold: float = 40.0,
    window_size: int = 5,
) -> list[Segment]:
    """Snap reference lyrics onto Whisper segment timestamps.

    Uses sequential fuzzy matching to preserve chronological order while correcting
    garbled words and hallucinations.
    """
    lyric_lines = parse_raw_lyrics(lyrics_text)
    if not lyric_lines or not segments:
        return segments

    norm_lyrics = [_normalize_lyric_line(l) for l in lyric_lines]
    out_segs: list[Segment] = []
    cur_lyric_idx = 0
    n_lyrics = len(lyric_lines)

    for seg in segments:
        seg_norm = _normalize_lyric_line(seg.text)
        if not seg_norm:
            out_segs.append(seg)
            continue

        best_score = 0.0
        best_idx = -1
        best_span = 1

        # Search window from cur_lyric_idx
        search_end = min(n_lyrics, cur_lyric_idx + window_size)
        for i in range(cur_lyric_idx, search_end):
            # Check single line
            score1 = fuzz.token_set_ratio(seg_norm, norm_lyrics[i])
            if score1 > best_score:
                best_score = score1
                best_idx = i
                best_span = 1

            # Check combined 2 lines if segment is long
            if i + 1 < n_lyrics:
                combined_norm = norm_lyrics[i] + " " + norm_lyrics[i + 1]
                score2 = fuzz.token_set_ratio(seg_norm, combined_norm)
                if score2 > best_score:
                    best_score = score2
                    best_idx = i
                    best_span = 2

        if best_score >= threshold and best_idx != -1:
            # Matched! Replace segment text with reference lyrics
            if best_span == 1:
                snapped_text = lyric_lines[best_idx]
            else:
                snapped_text = " ".join(lyric_lines[best_idx : best_idx + best_span])
            out_segs.append(replace(seg, text=snapped_text))
            cur_lyric_idx = best_idx + best_span
        else:
            # No strong match, keep original Whisper segment
            out_segs.append(seg)

    return out_segs
