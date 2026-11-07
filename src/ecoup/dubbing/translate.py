"""Translation with duration budget and rewriting loop (S5)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .asr import DubbingSegment


@dataclass
class TranslatedDubSegment:
    id: int
    start: float
    end: float
    duration: float
    speaker_id: str
    original_text: str
    translated_text: str
    syllable_count: int
    max_budget: int
    rewritten_times: int = 0
    overflow_flag: bool = False


def count_vietnamese_syllables(text: str) -> int:
    """In Vietnamese, syllables correspond directly to space-separated words."""
    words = [w for w in re.split(r"[\s\.,!\?\"\:\;]+", text.strip()) if w]
    return len(words)


@dataclass
class TranslationConfig:
    model: str = "gemini-2.5-flash"
    syllables_per_second: float = 3.8
    max_rewrite_attempts: int = 2
    temperature: float = 0.3
    mock: bool = False


class DubbingTranslator:
    def __init__(self, config: Optional[TranslationConfig] = None, api_key: Optional[str] = None):
        self.config = config or TranslationConfig()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            colab_key_p = Path("/content/.gemini_key")
            if colab_key_p.exists():
                try:
                    self.api_key = colab_key_p.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        if not self.api_key and not self.config.mock:
            # Check secret Colab key location
            if os.path.exists("/content/.gemini_key"):
                with open("/content/.gemini_key", "r", encoding="utf-8") as f:
                    self.api_key = f.read().strip()

    def translate_segments(
        self,
        segments: Sequence[DubbingSegment],
        context_notes: str = "",
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> list[TranslatedDubSegment]:
        if not segments:
            return []

        if self.config.mock:
            # Return deterministic mock translations
            mock_results = []
            for s in segments:
                budget = max(2, int(s.duration * self.config.syllables_per_second))
                vi = f"Dịch câu thoại mẫu {s.id}"
                s_count = count_vietnamese_syllables(vi)
                mock_results.append(
                    TranslatedDubSegment(
                        id=s.id,
                        start=s.start,
                        end=s.end,
                        duration=s.duration,
                        speaker_id=s.speaker_id or "spk_0",
                        original_text=s.text,
                        translated_text=vi,
                        syllable_count=s_count,
                        max_budget=budget,
                        rewritten_times=0,
                        overflow_flag=False,
                    )
                )
            return mock_results

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set or provided.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        prompt_payload = []
        for s in segments:
            budget = max(2, int(s.duration * self.config.syllables_per_second))
            prompt_payload.append({
                "id": s.id,
                "speaker": s.speaker_id or "spk_0",
                "en": s.text,
                "duration_s": s.duration,
                "max_syllables_budget": budget,
            })

        system_instruction = (
            "Bạn là biên kịch lồng tiếng phim chuyên nghiệp (Dubbing Scriptwriter). "
            "Nhiệm vụ: Dịch lời thoại tiếng Anh sang tiếng Việt tự nhiên, giữ phong cách hài hước và tính cách nhân vật. "
            "RÀNG BUỘC SỐNG CÒN: Số âm tiết tiếng Việt (words) của mỗi câu TUYỆT ĐỐI KHÔNG ĐƯỢC VƯỢT QUÁ max_syllables_budget "
            "để diễn viên lồng tiếng nói kịp khớp khung hình. Hãy dùng câu ngắn gọn, súc tích, thoát ý. "
            "Trả về JSON Array gồm các object: id, vi_text."
        )
        if context_notes:
            system_instruction += f"\nNgữ cảnh phim: {context_notes}"

        # 1. First translation pass in batches of 25 segments
        translated_map: dict[int, str] = {}
        batch_size = 25

        for b_idx in range(0, len(segments), batch_size):
            batch = segments[b_idx : b_idx + batch_size]
            prompt_payload = []
            for s in batch:
                budget = max(2, int(s.duration * self.config.syllables_per_second))
                prompt_payload.append({
                    "id": s.id,
                    "speaker": s.speaker_id or "spk_0",
                    "en": s.text,
                    "duration_s": s.duration,
                    "max_syllables_budget": budget,
                })

            config_args = {
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "temperature": self.config.temperature,
            }
            if hasattr(types, "ThinkingConfig"):
                config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            resp = client.models.generate_content(
                model=self.config.model,
                contents=f"Dịch danh sách câu sau sang tiếng Việt lồng tiếng:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}",
                config=types.GenerateContentConfig(**config_args),
            )

            try:
                items = json.loads(resp.text)
                for it in items:
                    val = it.get("vi_text") or it.get("vi") or it.get("vietnamese") or ""
                    translated_map[int(it["id"])] = str(val).strip()
            except Exception:
                pass

        # 2. Check budgets and rewrite if overflowed
        results: list[TranslatedDubSegment] = []
        need_rewrite: list[dict[str, Any]] = []

        for s in segments:
            budget = max(2, int(s.duration * self.config.syllables_per_second))
            vi = translated_map.get(s.id, s.text)
            s_count = count_vietnamese_syllables(vi)
            overflow = s_count > int(budget * 1.15)

            seg_res = TranslatedDubSegment(
                id=s.id,
                start=s.start,
                end=s.end,
                duration=s.duration,
                speaker_id=s.speaker_id or "spk_0",
                original_text=s.text,
                translated_text=vi,
                syllable_count=s_count,
                max_budget=budget,
                rewritten_times=0,
                overflow_flag=overflow,
            )
            results.append(seg_res)
            if overflow:
                need_rewrite.append({
                    "id": s.id,
                    "current_vi": vi,
                    "current_syllables": s_count,
                    "target_budget": budget,
                })

        # 3. Rewriting loop for overflowed sentences
        if need_rewrite and self.config.max_rewrite_attempts > 0:
            rewrite_prompt = (
                "Các câu thoại sau bị dài hơn thời lượng cho phép. Hãy viết lại câu ngắn hơn nữa (rút gọn từ ngữ, "
                "dùng từ đơn thay từ ghép) để số âm tiết <= target_budget nhưng vẫn giữ được ý chính:\n"
                f"{json.dumps(need_rewrite, ensure_ascii=False, indent=2)}\n"
                "Trả về JSON Array: id, vi_text."
            )
            try:
                rw_args = {
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                }
                if hasattr(types, "ThinkingConfig"):
                    rw_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

                rw_resp = client.models.generate_content(
                    model=self.config.model,
                    contents=rewrite_prompt,
                    config=types.GenerateContentConfig(**rw_args),
                )
                rw_items = json.loads(rw_resp.text)
                for it in rw_items:
                    r_id = int(it["id"])
                    new_vi = it.get("vi_text", "").strip()
                    target_seg = next((r for r in results if r.id == r_id), None)
                    if target_seg and new_vi:
                        new_count = count_vietnamese_syllables(new_vi)
                        target_seg.translated_text = new_vi
                        target_seg.syllable_count = new_count
                        target_seg.rewritten_times += 1
                        target_seg.overflow_flag = new_count > int(target_seg.max_budget * 1.15)
            except Exception:
                pass

        return results
