"""Gradio Web Application for Eco Video Upscaler (Optimized for Google Colab and remote devices).

Giao diện phục chế và nâng cấp độ phân giải video AI:
- Thiết kế tối giản, tập trung vào công việc, bố cục 2 cột chuyên nghiệp
- Hỗ trợ mô hình SeedVR2-3B DiT, CCSR v2 và các mô hình ONNX
- Hỗ trợ xem trước nhanh 30 khung (Quick Preview) có kèm âm thanh
- Hỗ trợ so sánh trực quan Bộ lọc tiền xử lý (Before / After Image Preview)
- Hỗ trợ hủy tiến trình tức thì và tự động Checkpoint / Resume
- Giám sát trạng thái và tài nguyên VRAM thời gian thực
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

# Prevent CUDA memory fragmentation on Colab Tesla T4 / L4
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Guard against diffusers / torchao version incompatibility (FqnToConfig missing in torchao 0.10)
try:
    import torchao.quantization
    if not hasattr(torchao.quantization, "FqnToConfig"):
        class FqnToConfig:
            pass
        torchao.quantization.FqnToConfig = FqnToConfig
except Exception:
    pass

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "seedvr_core"))

# Avoid UnicodeEncodeError on Windows consoles (cp1252)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ecoup.config import (
    AppConfig,
    ModelConfig,
    ThermalConfig,
    FrameSavingConfig,
    CheckpointConfig,
    EncodeConfig,
    InferenceConfig,
    PreProcessConfig,
    PostProcessConfig,
)
from ecoup.engine.onnx_engine import OnnxUpscaleEngine
from ecoup.pipeline import UpscalePipeline
from ecoup.utils import get_gpu_vram_info
from ecoup.video_io import get_video_metadata

# Global tracker for cancel functionality and engine cache
active_pipeline: Optional[UpscalePipeline] = None
active_subtitle_pipeline = None
_GLOBAL_ENGINE = None
_GLOBAL_ENGINE_KEY: Optional[str] = None

# Verified ONNX models mapping: model_key -> (filename, download_url, default_scale)
MODEL_REGISTRY = {
    "nomosuni": (
        "2xNomosUni_compact_otf_medium.onnx",
        "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2xNomosUni_compact_otf_medium.onnx",
        2,
    ),
    "x2plus": (
        "2x-realesrgan-x2plus.onnx",
        "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2x-realesrgan-x2plus.onnx",
        2,
    ),
    "general": (
        "realesr-general-x4v3.onnx",
        "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
        4,
    ),
    "anime": (
        "realesr-animevideov3.onnx",
        "https://huggingface.co/nimjinwei/designerplatform-models/resolve/main/realesr-animevideov3.onnx",
        4,
    ),
    "span": (
        "2x-spanx2-ch48.onnx",
        "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2x-spanx2-ch48.onnx",
        2,
    ),
}


def get_available_models():
    return [
        "SeedVR2-3B (ByteDance DiT Video)",
        "CCSR v2 (Diffusion Model)",
        "RealESRGAN x2plus (2x)",
        "NomosUni 2x",
        "RealESRGAN 4x (General)",
        "RealESRGAN 4x (Anime)",
        "SPAN 2x",
    ]


def ensure_model_file(model_key: str) -> tuple[Path, int]:
    """Ensure ONNX model file exists on disk, downloading if necessary."""
    if model_key not in MODEL_REGISTRY:
        model_key = "nomosuni"
    filename, url, scale = MODEL_REGISTRY[model_key]
    dest = Path("models") / filename
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Model Loader] Đang tải mô hình {filename} từ {url}...")
        urllib.request.urlretrieve(url, str(dest))
    return dest, scale


def cancel_upscale():
    """Cancel currently running upscale job."""
    global active_pipeline
    if active_pipeline is not None:
        active_pipeline.cancel()
        return "⏹️ Đã gửi lệnh hủy. Đang dừng an toàn và giải phóng GPU..."
    return "Không có tiến trình nào đang chạy."


def cancel_subtitle():
    """Cancel currently running subtitle job."""
    global active_subtitle_pipeline
    if active_subtitle_pipeline is not None:
        active_subtitle_pipeline.cancel()
        return "⏹️ Đã gửi lệnh hủy tiến trình phụ đề."
    return "Không có tiến trình phụ đề nào đang chạy."


def inspect_video(file_path: Optional[str]) -> str:
    """Extract metadata using ffprobe and format for UI display."""
    if not file_path or not os.path.exists(file_path):
        return "ℹ️ *Chưa chọn tệp video.*"
    try:
        m = get_video_metadata(file_path)
        warn = ""
        if m.width < 720 or m.height < 720:
            warn = "<br>⚠️ *Nguồn độ phân giải thấp, khuyên dùng SeedVR2 để phục chế chi tiết.*"
        audio_info = f"Có ({m.audio_codec})" if m.has_audio else "Không có audio"
        return (
            f"📹 **Độ phân giải:** `{m.width}×{m.height}` • **FPS:** `{m.fps:.2f}` ({m.fps_fraction})<br>"
            f"⏱️ **Thời lượng:** `{m.duration_seconds:.2f}s` (`{m.total_frames}` khung hình)<br>"
            f"🔊 **Âm thanh:** `{audio_info}` • **Codec Video:** `{m.video_codec}`{warn}"
        )
    except Exception as e:
        return f"Lỗi đọc thông tin: {e}"


def get_hardware_status() -> str:
    """Detect GPU hardware and return formatted status badge."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_gb = total_bytes / (1024 ** 3)
            free_gb = free_bytes / (1024 ** 3)
            return f"🟢 **GPU:** {gpu_name} • **VRAM:** `{free_gb:.1f}GB trống / {total_gb:.1f}GB tổng`"
    except Exception:
        pass
    return "🟡 **GPU:** CPU Mode / CUDA chưa sẵn sàng."


def run_upscale(
    input_file_path: Optional[str],
    model_choice: str,
    target_scale_choice: str,
    tile_choice: str,
    control_strength: float,
    denoise_method_choice: str,
    pre_denoise_pct: int,
    recover_detail_pct: int,
    sharpen_pct: int,
    film_grain_checked: bool,
    memory_mode_choice: str,
    sub_len_choice: int,
    model_res_choice: str,
    seed_input: int,
    random_seed_checked: bool,
    checkpoint_enabled: bool,
    resume_checked: bool,
    preview_mode: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    """Execute video upscaling pipeline."""
    global active_pipeline

    if not input_file_path or not os.path.exists(input_file_path):
        raise gr.Error("Vui lòng chọn video đầu vào hợp lệ.")

    in_path = Path(input_file_path)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if preview_mode:
        out_path = output_dir / f"{in_path.stem}_preview.mp4"
    else:
        out_path = output_dir / f"{in_path.stem}_upscaled.mp4"

    # Scale mapping (Phase 4.1)
    if "1.5" in str(target_scale_choice):
        target_scale = 1.5
    elif "4" in str(target_scale_choice):
        target_scale = 4.0
    else:
        target_scale = 2.0

    # Memory mode mapping
    mem_mode = "auto"
    if "resident" in memory_mode_choice.lower():
        mem_mode = "resident"
    elif "offload" in memory_mode_choice.lower():
        mem_mode = "offload"

    # Model resolution mapping
    model_res = "match_target" if "match" in model_res_choice.lower() else "native"

    # Tile mapping
    if "toàn khung" in tile_choice.lower() or "full" in tile_choice.lower():
        tile_sz = 0
        tile_ov = 0
    elif "256" in tile_choice:
        tile_sz = 256
        tile_ov = 16
    else:  # 512
        tile_sz = 512
        tile_ov = 16

    progress(0.01, desc="Đang chuẩn bị mô hình AI...")

    global _GLOBAL_ENGINE, _GLOBAL_ENGINE_KEY

    # Clean unreferenced GPU memory before starting
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    engine = None
    custom_fn = None
    model_name = "nomosuni"
    model_scale = 2
    model_file_str = "models/model.onnx"

    used_seed = None if random_seed_checked else int(seed_input)

    if "seedvr" in model_choice.lower():
        from ecoup.engine.seedvr2_engine import SeedVR2UpscaleEngine, _configure_vae_for_colab, _clear_vae_memory
        model_name = "SeedVR2-3B"
        model_scale = 2
        model_file_str = "ckpts/seedvr2_ema_3b.pth"

        # Check if already loaded in memory
        if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE_KEY == "seedvr2" and isinstance(_GLOBAL_ENGINE, SeedVR2UpscaleEngine):
            engine = _GLOBAL_ENGINE
            engine.scale = float(target_scale)
            engine.seed = used_seed
            engine.sub_len = int(sub_len_choice)
            engine.model_resolution = model_res
            if engine.memory_mode != mem_mode:
                engine.memory_mode = mem_mode
                engine.effective_memory_mode = mem_mode
            if hasattr(engine, "runner") and engine.runner is not None and hasattr(engine.runner, "vae"):
                _configure_vae_for_colab(engine.runner.vae)
                _clear_vae_memory(engine.runner.vae)
            print("[SeedVR2 App] Đã tái sử dụng mô hình SeedVR2 sẵn có trên bộ nhớ.")
        else:
            if _GLOBAL_ENGINE is not None:
                del _GLOBAL_ENGINE
                _GLOBAL_ENGINE = None
                _GLOBAL_ENGINE_KEY = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            engine = SeedVR2UpscaleEngine(
                scale=float(target_scale),
                precision="fp16",
                seed=used_seed,
                memory_mode=mem_mode,
                sub_len=int(sub_len_choice),
                model_resolution=model_res,
            )
            engine.load(model_path="ckpts/seedvr2_ema_3b.pth", precision="fp16", provider="cuda")
            _GLOBAL_ENGINE = engine
            _GLOBAL_ENGINE_KEY = "seedvr2"
    elif "ccsr" in model_choice.lower():
        if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE_KEY != "ccsr":
            del _GLOBAL_ENGINE
            _GLOBAL_ENGINE = None
            _GLOBAL_ENGINE_KEY = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        from ecoup.engine.ccsr_engine import CcsrUpscaleEngine
        engine = CcsrUpscaleEngine(control_strength=control_strength)
        model_name = "CCSR_v2"
        model_scale = 2
        engine.load(model_path="preset/models", precision="fp16", provider="cuda")
        model_file_str = "preset/models/controlnet"
        _GLOBAL_ENGINE = engine
        _GLOBAL_ENGINE_KEY = "ccsr"
    elif "x2plus" in model_choice.lower():
        if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE_KEY != "x2plus":
            del _GLOBAL_ENGINE
            _GLOBAL_ENGINE = None
            _GLOBAL_ENGINE_KEY = None
        mf, model_scale = ensure_model_file("x2plus")
        model_name = "RealESRGAN_x2plus"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "anime" in model_choice.lower():
        if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE_KEY != "anime":
            del _GLOBAL_ENGINE
            _GLOBAL_ENGINE = None
            _GLOBAL_ENGINE_KEY = None
        mf, model_scale = ensure_model_file("anime")
        model_name = "realesr-animevideov3"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "span" in model_choice.lower():
        if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE_KEY != "span":
            del _GLOBAL_ENGINE
            _GLOBAL_ENGINE = None
            _GLOBAL_ENGINE_KEY = None
        mf, model_scale = ensure_model_file("span")
        model_name = "SPAN_2x"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "general" in model_choice.lower():
        if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE_KEY != "general":
            del _GLOBAL_ENGINE
            _GLOBAL_ENGINE = None
            _GLOBAL_ENGINE_KEY = None
        mf, model_scale = ensure_model_file("general")
        model_name = "realesr-general-x4v3"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    else:
        if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE_KEY != "nomosuni":
            del _GLOBAL_ENGINE
            _GLOBAL_ENGINE = None
            _GLOBAL_ENGINE_KEY = None
        mf, model_scale = ensure_model_file("nomosuni")
        model_name = "NomosUni_2x"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")

    if engine is not None and hasattr(engine, "scale") and engine.scale > 0:
        model_scale = engine.scale

    # Checkpoint configuration (Phase 1.1)
    drive_chk = Path("/content/drive/MyDrive/ecoup_checkpoints")
    chk_dir = str(drive_chk) if drive_chk.parent.exists() else ".ecoup_state"

    cfg = AppConfig(
        model=ModelConfig(
            name=model_name,
            path=model_file_str,
            scale=model_scale,
            target_scale=target_scale,
            precision="fp16",
        ),
        inference=InferenceConfig(
            provider="auto",
            tile_size=tile_sz,
            tile_overlap=tile_ov,
            control_strength=control_strength,
            clip_length=int(sub_len_choice) if "seedvr" in model_choice.lower() else 16,
            clip_overlap=1 if "seedvr" in model_choice.lower() else 2,
            seed=used_seed,
            memory_mode=mem_mode,
            sub_len=int(sub_len_choice),
            model_resolution=model_res,
        ),
        thermal=ThermalConfig(enabled=False),
        frame_saving=FrameSavingConfig(
            enable_dedup=("ccsr" not in model_choice.lower() and "seedvr" not in model_choice.lower())
        ),
        checkpoint=CheckpointConfig(
            enabled=checkpoint_enabled and not preview_mode,
            dir=chk_dir,
            interval_frames=100,
        ),
        encode=EncodeConfig(codec="auto", copy_audio=True),
        pre_process=PreProcessConfig(
            method="none" if "none" in denoise_method_choice.lower() else ("nlmeans" if "nlmeans" in denoise_method_choice.lower() else "bilateral"),
            denoise=pre_denoise_pct / 100.0,
        ),
        post_process=PostProcessConfig(
            color_fix=True,
            recover_detail=recover_detail_pct / 100.0,
            sharpen=sharpen_pct / 100.0,
            film_grain=0.015 if film_grain_checked else 0.0,
        ),
    )

    last_status = "Đang bắt đầu xử lý..."

    def on_progress(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate):
        nonlocal last_status
        pct = (frame_idx / max(1, total_frames))
        mins = int(eta_sec // 60)
        secs = int(eta_sec % 60)

        used_gb, total_gb, gpu_name = get_gpu_vram_info()
        vram_info = ""
        if total_gb > 0:
            vram_pct = (used_gb / total_gb) * 100
            vram_info = f" • VRAM: {used_gb:.1f}/{total_gb:.1f}GB ({vram_pct:.0f}%)"

        prov = getattr(engine, "active_provider", "GPU")

        last_status = (
            f"Khung hình {frame_idx}/{total_frames} ({int(pct * 100)}%) "
            f"• {fps:.2f} fps • Còn lại: {mins:02d}:{secs:02d}{vram_info} • {prov}"
        )
        progress(pct, desc=last_status)

    pipeline = UpscalePipeline(
        config=cfg,
        engine=engine,
        custom_upscale_fn=custom_fn,
        on_progress=on_progress,
    )
    active_pipeline = pipeline

    # Handle preview mode (Phase 4.4)
    run_input_path = in_path
    if preview_mode:
        preview_in = output_dir / f"{in_path.stem}_preview_raw.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(in_path),
            "-vframes", "30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(preview_in)
        ]
        try:
            subprocess.run(cmd, check=True)
            run_input_path = preview_in
        except Exception:
            run_input_path = in_path

    try:
        progress(0.02, desc="Đang phân tích video...")
        pipeline.run(run_input_path, out_path, resume=(resume_checked and not preview_mode))
    except Exception as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"[Colab App] Phát hiện OOM: {e}. Đang giải phóng bộ nhớ...")
            _GLOBAL_ENGINE = None
            _GLOBAL_ENGINE_KEY = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if pipeline._cancelled:
            return None, "⏹️ Quá trình xử lý đã được hủy thành công."
        raise e
    finally:
        active_pipeline = None
        if _GLOBAL_ENGINE is not None and hasattr(_GLOBAL_ENGINE, "runner") and _GLOBAL_ENGINE.runner is not None:
            if hasattr(_GLOBAL_ENGINE.runner, "vae"):
                from ecoup.engine.seedvr2_engine import _clear_vae_memory
                _clear_vae_memory(_GLOBAL_ENGINE.runner.vae)
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if pipeline._cancelled:
        return None, "⏹️ Quá trình xử lý đã được hủy thành công."

    progress(1.0, desc="Hoàn tất xử lý.")
    tag = "Xem thử (30 khung)" if preview_mode else "Toàn bộ"
    return str(out_path), f"🎉 Hoàn tất ({tag})! Video đã lưu: {out_path.name}"


def run_transcribe_only(
    input_file_path: Optional[str],
    source_lang_choice: str,
    audio_mode_choice: str,
    whisper_model_choice: str,
    initial_prompt_input: str,
    use_demucs_choice: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Run ASR only and return original SRT text for preview and manual editing."""
    global active_subtitle_pipeline

    if not input_file_path or not os.path.exists(input_file_path):
        raise gr.Error("Vui lòng chọn video đầu vào hợp lệ.")

    from ecoup.subtitle.config import SubtitleConfig
    from ecoup.subtitle.pipeline import SubtitlePipeline

    cfg = SubtitleConfig()
    if audio_mode_choice:
        mode_token = audio_mode_choice.split()[0].strip().lower()
        if mode_token in ("music", "speech", "off"):
            cfg.asr.audio_mode = mode_token
    if not source_lang_choice or "auto" in source_lang_choice.lower():
        cfg.asr.language = None
    else:
        cfg.asr.language = source_lang_choice.split()[0].strip()
    if whisper_model_choice:
        cfg.asr.model = whisper_model_choice.split()[0].strip()
    if initial_prompt_input and initial_prompt_input.strip():
        cfg.asr.initial_prompt = initial_prompt_input.strip()
    cfg.asr.use_demucs = bool(use_demucs_choice)

    out_dir = Path("outputs") / "subtitles"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress(0.1, desc="Khởi động bộ nhận diện giọng nói...")

    pipe = SubtitlePipeline(config=cfg)
    active_subtitle_pipeline = pipe
    try:
        progress(0.4, desc="Đang nhận dạng văn bản bằng Whisper...")
        srt_path, srt_content, rep = pipe.transcribe_only(input_file_path, out_dir=out_dir)
    except Exception as e:
        if pipe._cancelled:
            return "", "⏹️ Đã hủy tiến trình nhận diện."
        raise gr.Error(f"Lỗi nhận diện âm thanh: {e}") from e
    finally:
        active_subtitle_pipeline = None

    msg = f"✅ Nhận diện hoàn tất! Đã tìm thấy {rep.cues} câu ({rep.detected_language})."
    if rep.demucs_applied:
        msg += " [Đã tách giọng hát Demucs 44.1kHz]"
    elif use_demucs_choice and any("Demucs" in w for w in rep.warnings):
        msg += " [⚠️ Cảnh báo: Demucs thất bại, đã dùng âm thanh gốc]"
    msg += " Bạn có thể sửa trực tiếp ở ô bên dưới."
    return srt_content, msg


def run_subtitle(
    input_file_path: Optional[str],
    source_lang_choice: str,
    target_lang_choice: str,
    audio_mode_choice: str,
    whisper_model_choice: str,
    gemini_model_choice: str,
    api_key_input: str,
    context_note_input: str,
    glossary_input: str,
    burn_in_checked: bool,
    show_orig_checked: bool,
    font_name_input: str,
    initial_prompt_input: str = "",
    custom_srt_input: str = "",
    use_demucs_choice: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    """Execute automatic subtitle pipeline (Audio extract -> Whisper ASR -> Gemini Translate -> Burn-in)."""
    global active_subtitle_pipeline

    if not input_file_path or not os.path.exists(input_file_path):
        raise gr.Error("Vui lòng chọn video đầu vào hợp lệ.")

    # API key resolution
    resolved_key = (api_key_input or "").strip()
    if not resolved_key:
        resolved_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not resolved_key:
        try:
            from google.colab import userdata
            resolved_key = (userdata.get("GEMINI_API_KEY") or "").strip()
        except Exception:
            pass
    if not resolved_key and Path("/content/.gemini_key").exists():
        try:
            resolved_key = Path("/content/.gemini_key").read_text(encoding="utf-8").strip()
        except Exception:
            pass

    if resolved_key:
        os.environ["GEMINI_API_KEY"] = resolved_key
    elif not (source_lang_choice and source_lang_choice.startswith("vi")):
        raise gr.Error(
            "⚠️ Chưa tìm thấy Gemini API Key! Vui lòng dán khóa API vào ô 'Khóa Gemini API' trên giao diện (hoặc cấu hình trong Colab Secrets: GEMINI_API_KEY)."
        )

    from ecoup.subtitle.config import SubtitleConfig
    from ecoup.subtitle.pipeline import SubtitlePipeline

    cfg = SubtitleConfig()

    # Audio mode & VAD resolution
    if audio_mode_choice:
        mode_token = audio_mode_choice.split()[0].strip().lower()
        if mode_token in ("music", "speech", "off"):
            cfg.asr.audio_mode = mode_token

    # Source language resolution
    if not source_lang_choice or "auto" in source_lang_choice.lower():
        cfg.asr.language = None
    else:
        cfg.asr.language = source_lang_choice.split()[0].strip()

    cfg.translate.target_language = target_lang_choice.split()[0].strip() if target_lang_choice else "vi"

    # Models
    if whisper_model_choice:
        cfg.asr.model = whisper_model_choice.split()[0].strip()
    if gemini_model_choice:
        cfg.translate.model = gemini_model_choice.split()[0].strip()
    if initial_prompt_input and initial_prompt_input.strip():
        cfg.asr.initial_prompt = initial_prompt_input.strip()
    cfg.asr.use_demucs = bool(use_demucs_choice)

    # Context note & Glossary
    cfg.translate.context_note = context_note_input.strip() if context_note_input else ""
    glossary = {}
    if glossary_input and glossary_input.strip():
        text = glossary_input.strip()
        if text.startswith("{"):
            try:
                import json
                glossary = json.loads(text)
            except Exception:
                pass
        if not glossary:
            for line in text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    glossary[k.strip()] = v.strip()
                elif "=" in line:
                    k, v = line.split("=", 1)
                    glossary[k.strip()] = v.strip()
    cfg.translate.glossary = glossary

    # Cache & outputs
    out_dir = Path("outputs") / "subtitles"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.translate.cache_dir = str(out_dir / "cache")
    cfg.burn_in = bool(burn_in_checked)
    cfg.burn.show_original = bool(show_orig_checked)
    if font_name_input and font_name_input.strip():
        cfg.burn.font_name = font_name_input.strip()

    step_labels = {
        "audio": ("Đang tách luồng âm thanh bằng FFmpeg...", 0.10),
        "asr": ("Đang nhận dạng giọng nói bằng Whisper...", 0.40),
        "normalize": ("Đang chuẩn hóa câu và mốc thời gian...", 0.50),
        "translate": ("Đang dịch sang tiếng Việt bằng Gemini...", 0.80),
        "srt": ("Đang xuất tệp phụ đề SRT...", 0.85),
        "burn": ("Đang ghép phụ đề cứng vào video (FFmpeg libass)...", 0.95),
    }

    def on_sub_progress(step: str, frac: float):
        desc, base_pct = step_labels.get(step, (f"Đang xử lý {step}...", 0.5))
        progress(base_pct, desc=desc)

    pipe = SubtitlePipeline(config=cfg, on_progress=on_sub_progress)
    active_subtitle_pipeline = pipe

    try:
        progress(0.02, desc="Khởi động pipeline phụ đề...")
        in_p = Path(input_file_path)
        out_video_path = out_dir / f"{in_p.stem}_sub.mp4"
        custom_srt_val = custom_srt_input.strip() if custom_srt_input and custom_srt_input.strip() else None
        rep = pipe.run(input_file_path, output=out_video_path, out_dir=out_dir, custom_srt=custom_srt_val)
    except Exception as e:
        if pipe._cancelled:
            return None, None, None, None, None, "⏹️ Tiến trình phụ đề đã được hủy thành công."
        raise gr.Error(f"Lỗi tạo phụ đề: {e}") from e
    finally:
        active_subtitle_pipeline = None

    if rep.cancelled:
        return None, None, None, None, None, "⏹️ Tiến trình phụ đề đã được hủy thành công."

    progress(1.0, desc="Hoàn tất tạo phụ đề!")
    status_msg = (
        f"🎉 Hoàn tất tạo phụ đề! Ngôn ngữ phát hiện: {rep.detected_language} "
        f"(Độ tin cậy: {rep.language_probability * 100:.0f}%). "
        f"Tổng {rep.cues} câu thoại. Thời gian xử lý: {rep.timings_s.get('total', 0)}s."
    )
    if rep.demucs_applied:
        status_msg += " (Đã tách giọng hát Demucs 44.1kHz)"
    elif use_demucs_choice and any("Demucs" in w for w in rep.warnings):
        status_msg += " (⚠️ Cảnh báo: Demucs thất bại, đã dùng âm thanh gốc)"
    if rep.skipped_translation:
        status_msg += " (Bỏ qua dịch do video nguồn đã là tiếng Việt)"
    elif rep.translated_failed:
        status_msg += f" (Lưu ý: {rep.translated_failed} câu giữ nguyên do lỗi/chặn)"

    return (
        rep.outputs.get("video"),
        rep.outputs.get("srt_translated"),
        rep.outputs.get("srt_original"),
        rep.outputs.get("ass"),
        str(out_dir / f"{in_p.stem}.subtitle_report.json"),
        status_msg,
    )


def create_ui():
    """Create clean, modern Gradio interface."""
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
    #header { text-align: center; margin-bottom: 12px; }
    #header h1 { font-size: 2.0rem; font-weight: 800; margin-bottom: 4px; background: linear-gradient(90deg, #2563eb, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    #header p { color: #64748b; font-size: 0.95rem; margin: 0; }
    .gr-button-primary { background: linear-gradient(90deg, #2563eb, #1d4ed8) !important; color: white !important; font-weight: 600 !important; }
    .gr-button-secondary { background: #0284c7 !important; color: white !important; font-weight: 600 !important; }
    .gr-button-stop { background: #dc2626 !important; color: white !important; font-weight: 600 !important; }
    """

    with gr.Blocks(title="Eco Video Upscaler") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        with gr.Column(elem_id="header"):
            gr.Markdown(
                """
                # ⚡ Eco Video Upscaler
                Hệ thống phục chế và nâng cấp độ phân giải video AI (ByteDance SeedVR2-3B DiT, CCSR v2 & ONNX)
                """
            )
            hw_badge = gr.Markdown(get_hardware_status())

        with gr.Tabs():
            with gr.Tab("⚡ Phục chế & Siêu phân giải (Upscale AI)"):
                with gr.Row():
                    # ================= LEFT COLUMN: INPUTS & SETTINGS =================
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("### 1. Tệp nguồn & Cấu hình cơ bản")
                            in_file = gr.File(
                                label="Tải lên Video nguồn",
                                file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm"],
                                type="filepath",
                            )
                            video_info_md = gr.Markdown("ℹ️ *Chưa chọn tệp video.*")
                            in_file.change(fn=inspect_video, inputs=[in_file], outputs=[video_info_md])
        
                            sample_videos = [
                                "/content/upscale_video/plan_for_project/tests_video/720_A.mp4",
                                "/content/upscale_video/plan_for_project/tests_video/720_B.mp4",
                                "/content/upscale_video/plan_for_project/tests_video/720_C.mp4",
                            ]
                            existing_samples = [[s] for s in sample_videos if Path(s).exists()]
                            if existing_samples:
                                gr.Examples(
                                    examples=existing_samples,
                                    inputs=[in_file],
                                    label="Video mẫu có sẵn trên máy chủ (Bấm để nạp nhanh)",
                                )
        
                            with gr.Row():
                                model_dd = gr.Dropdown(
                                    choices=get_available_models(),
                                    value="SeedVR2-3B (ByteDance DiT Video)",
                                    label="Mô hình AI",
                                    scale=3,
                                )
                                scale_dd = gr.Dropdown(
                                    choices=[
                                        "1.5x (1080p Dọc - 1080×1920)",
                                        "2x (1440p Dọc - 1440×2560)",
                                        "4x",
                                    ],
                                    value="2x (1440p Dọc - 1440×2560)",
                                    label="Tỉ lệ phóng đại",
                                    scale=2,
                                )
                                tile_dd = gr.Dropdown(
                                    choices=[
                                        "Toàn khung (Full-Frame)",
                                        "Tile 512x512",
                                        "Tile 256x256",
                                    ],
                                    value="Toàn khung (Full-Frame)",
                                    label="Phân mảnh (Tile)",
                                    scale=2,
                                )
        
                        with gr.Accordion("🎨 2. Bộ lọc tiền xử lý & Màu sắc (Enhancements)", open=False):
                            with gr.Row():
                                denoise_method_dd = gr.Dropdown(
                                    choices=[
                                        "bilateral (Lọc cạnh biên - Khuyên dùng)",
                                        "nlmeans (Lọc chi tiết)",
                                        "none (Không lọc)",
                                    ],
                                    value="bilateral (Lọc cạnh biên - Khuyên dùng)",
                                    label="Phương pháp khử nhiễu",
                                )
                                denoise_sl = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=10,
                                    step=5,
                                    label="Khử nhiễu (Denoise %)",
                                )
                                recover_sl = gr.Slider(
                                    minimum=0,
                                    maximum=50,
                                    value=10,
                                    step=5,
                                    label="Khôi phục gốc (Recover %)",
                                )
        
                            with gr.Row():
                                ctrl_strength_sl = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.5,
                                    value=0.75,
                                    step=0.05,
                                    label="Độ ép cấu trúc (ControlNet)",
                                )
                                sharpen_sl = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=15,
                                    step=5,
                                    label="Làm nét (Sharpen %)",
                                )
                                grain_cb = gr.Checkbox(
                                    value=False,
                                    label="Bù hạt phim (Film Grain)",
                                )
        
                        with gr.Accordion("⚙️ 3. Cấu hình Nâng cao GPU & SeedVR2", open=False):
                            with gr.Row():
                                memory_mode_dd = gr.Dropdown(
                                    choices=[
                                        "auto (Tự chọn theo VRAM - Khuyên dùng)",
                                        "resident (Nhanh nhất 2.4x - Giữ GPU)",
                                        "offload (Bảo vệ VRAM 16GB)",
                                    ],
                                    value="auto (Tự chọn theo VRAM - Khuyên dùng)",
                                    label="Chế độ bộ nhớ (Memory Mode)",
                                )
                                sub_len_dd = gr.Dropdown(
                                    choices=[5, 9, 13],
                                    value=5,
                                    label="Cửa sổ thời gian DiT (Temporal Window)",
                                )
                                model_res_dd = gr.Dropdown(
                                    choices=[
                                        "native (Gốc 720p - Ổn định cao)",
                                        "match_target (Khớp độ phân giải phóng đại - Cần GPU >40GB)",
                                    ],
                                    value="native (Gốc 720p - Ổn định cao)",
                                    label="Độ phân giải nội bộ DiT",
                                )
        
                            with gr.Row():
                                seed_num = gr.Number(
                                    value=42,
                                    label="Seed cố định (Khử nhấp nháy chi tiết)",
                                    precision=0,
                                )
                                random_seed_cb = gr.Checkbox(
                                    value=False,
                                    label="Dùng Seed ngẫu nhiên (None)",
                                )
                                checkpoint_cb = gr.Checkbox(
                                    value=True,
                                    label="Bật Checkpoint lưu phân đoạn",
                                )
                                resume_cb = gr.Checkbox(
                                    value=False,
                                    label="Tiếp tục job dở (Resume)",
                                )
        
                    # ================= RIGHT COLUMN: CONTROLS & OUTPUT =================
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("### 4. Điều khiển thực thi")
                            with gr.Row():
                                btn_preview = gr.Button("⚡ Xem thử nhanh (30 khung)", variant="secondary", size="lg", scale=2)
                                btn_run = gr.Button("🚀 Bắt đầu xử lý toàn bộ", variant="primary", size="lg", scale=3)
                                btn_cancel = gr.Button("⏹️ Hủy", variant="stop", size="lg", scale=1)
        
                            status_output = gr.Textbox(
                                label="Trạng thái xử lý",
                                value="Sẵn sàng",
                                interactive=False,
                                lines=1,
                            )
        
                        with gr.Group():
                            gr.Markdown("### 5. Kết quả Video thành phẩm")
                            out_file = gr.File(
                                label="Video thành phẩm (Tải về)",
                                interactive=False,
                            )
        
                        with gr.Accordion("💡 Hướng dẫn & Lưu ý phần cứng", open=False):
                            gr.Markdown(
                                """
                                - **Tesla T4 (15GB VRAM)**: Khuyên dùng `memory_mode: offload`, `sub_len: 5`, `model_resolution: native`.
                                - **NVIDIA L4 (24GB VRAM)**: Khuyên dùng `memory_mode: resident` (tăng tốc gấp **2.4 lần**, đạt ~0.35 - 0.50 FPS).
                                - **NVIDIA A100 (40GB/80GB)**: Có thể tăng `sub_len: 9` hoặc `13` để tăng tính liên tục thời gian.
                                - **Bảo toàn số khung hình**: Hệ thống bảo toàn 100% số lượng khung hình gốc và tự động remux âm thanh AAC đồng bộ.
                                """
                            )
        
                common_inputs = [
                    in_file,
                    model_dd,
                    scale_dd,
                    tile_dd,
                    ctrl_strength_sl,
                    denoise_method_dd,
                    denoise_sl,
                    recover_sl,
                    sharpen_sl,
                    grain_cb,
                    memory_mode_dd,
                    sub_len_dd,
                    model_res_dd,
                    seed_num,
                    random_seed_cb,
                    checkpoint_cb,
                    resume_cb,
                ]
        
                def on_run_full(*args):
                    return run_upscale(*args, preview_mode=False)
        
                def on_run_preview(*args):
                    return run_upscale(*args, preview_mode=True)
        
                run_event = btn_run.click(
                    fn=on_run_full,
                    inputs=common_inputs,
                    outputs=[out_file, status_output],
                )
        
                preview_event = btn_preview.click(
                    fn=on_run_preview,
                    inputs=common_inputs,
                    outputs=[out_file, status_output],
                )
        
                btn_cancel.click(
                    fn=cancel_upscale,
                    outputs=[status_output],
                    cancels=[run_event, preview_event],
                )
        
            with gr.Tab("📝 Phụ đề tự động (Whisper + Gemini)"):
                with gr.Row():
                    # ================= LEFT COLUMN: SUBTITLE INPUTS =================
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("### 1. Tệp nguồn & Ngôn ngữ")
                            sub_in_file = gr.File(
                                label="Tải lên Video cần tạo phụ đề",
                                file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm"],
                                type="filepath",
                            )
                            sub_video_info_md = gr.Markdown("ℹ️ *Chưa chọn tệp video.*")
                            sub_in_file.change(fn=inspect_video, inputs=[sub_in_file], outputs=[sub_video_info_md])

                            sub_sample_videos = [
                                "/content/upscale_video/plan_for_project/tests_video/sub_test_multilingual_light.mp4",
                                "/content/upscale_video/plan_for_project/tests_video/sub_test_eng.mp4",
                                "/content/upscale_video/plan_for_project/tests_video/sub_test_cn.mp4",
                                "/content/upscale_video/plan_for_project/tests_video/sub_test_jp.mp4",
                                "/content/upscale_video/plan_for_project/tests_video/sub_test_kor.mp4",
                                "/content/upscale_video/plan_for_project/tests_video/sub_test_vie.mp4",
                            ]
                            sub_existing_samples = [[s] for s in sub_sample_videos if Path(s).exists()]
                            if sub_existing_samples:
                                gr.Examples(
                                    examples=sub_existing_samples,
                                    inputs=[sub_in_file],
                                    label="Video mẫu thử nghiệm phụ đề (Bấm để nạp nhanh)",
                                )

                            with gr.Row():
                                sub_source_lang_dd = gr.Dropdown(
                                    choices=[
                                        "auto (Tự động nhận diện)",
                                        "en (Tiếng Anh)",
                                        "zh (Tiếng Trung)",
                                        "ja (Tiếng Nhật)",
                                        "ko (Tiếng Hàn)",
                                        "vi (Tiếng Việt)",
                                        "fr (Tiếng Pháp)",
                                        "de (Tiếng Đức)",
                                        "ru (Tiếng Nga)",
                                        "th (Tiếng Thái)",
                                    ],
                                    value="auto (Tự động nhận diện)",
                                    label="Ngôn ngữ nguồn",
                                    scale=3,
                                )
                                sub_target_lang_dd = gr.Dropdown(
                                    choices=["vi (Tiếng Việt)"],
                                    value="vi (Tiếng Việt)",
                                    label="Ngôn ngữ đích",
                                    scale=2,
                                )

                            sub_audio_mode_dd = gr.Dropdown(
                                choices=[
                                    "speech (Hội thoại / Vlog / Phim - Lọc tiếng ồn chuẩn)",
                                    "music (Âm nhạc / MV / Beat dày - Bắt trọn lời bài hát)",
                                    "off (Tắt hoàn toàn bộ lọc VAD)",
                                ],
                                value="speech (Hội thoại / Vlog / Phim - Lọc tiếng ồn chuẩn)",
                                label="Chế độ âm thanh (VAD Audio Mode)",
                                info="Mặc định 'speech' cho vlog/hội thoại lọc ồn; chọn 'music' hoặc 'off' cho bài hát/MV để không bị nuốt lời.",
                            )
                            with gr.Row():
                                sub_use_demucs_cb = gr.Checkbox(
                                    value=False,
                                    label="Tách giọng hát bằng AI (Demucs Vocal Separation - Tối ưu cho bài hát/MV có beat lớn)",
                                )
                            sub_initial_prompt_tb = gr.Textbox(
                                label="Từ vựng gợi ý / Lời bài hát tham khảo (Initial Prompt)",
                                placeholder="Nhập tên riêng, thuật ngữ hoặc lời bài hát để Whisper bắt từ chuẩn xác (ví dụ: Tequila, formula, medulla...)...",
                                lines=2,
                            )

                        with gr.Group():
                            gr.Markdown("### 2. Mô hình AI & Khóa API")
                            with gr.Row():
                                sub_whisper_dd = gr.Dropdown(
                                    choices=[
                                        "large-v3 (Chính xác cao nhất - Khuyên dùng)",
                                        "medium",
                                        "small",
                                        "base",
                                    ],
                                    value="large-v3 (Chính xác cao nhất - Khuyên dùng)",
                                    label="Mô hình nhận dạng giọng nói (Whisper)",
                                    scale=3,
                                )
                                sub_gemini_dd = gr.Dropdown(
                                    choices=[
                                        "gemini-3.1-flash-lite (Nhanh & Tối ưu chi phí)",
                                        "gemini-3.5-flash-lite",
                                        "gemini-2.5-flash-lite",
                                        "gemini-3.5-flash",
                                    ],
                                    value="gemini-3.1-flash-lite (Nhanh & Tối ưu chi phí)",
                                    label="Mô hình dịch thuật (Gemini)",
                                    scale=3,
                                )

                            _pre_key = os.environ.get("GEMINI_API_KEY", "").strip()
                            if not _pre_key and Path("/content/.gemini_key").exists():
                                try:
                                    _pre_key = Path("/content/.gemini_key").read_text(encoding="utf-8").strip()
                                except Exception:
                                    pass
                            if not _pre_key:
                                try:
                                    from google.colab import userdata
                                    _pre_key = (userdata.get("GEMINI_API_KEY") or "").strip()
                                except Exception:
                                    pass

                            sub_api_key_tb = gr.Textbox(
                                label="Khóa Gemini API (✅ Đã tự động nạp từ Colab)" if _pre_key else "Khóa Gemini API (Dán vào đây hoặc thiết lập trong Colab Secrets)",
                                placeholder="Đã nạp sẵn API Key từ hệ thống (có thể để trống)" if _pre_key else "Dán API Key vào đây (ví dụ: AQ.Ab8... hoặc AIzaSy...)",
                                value=_pre_key if _pre_key else "",
                                type="password",
                            )

                        with gr.Accordion("📝 3. Tinh chỉnh ngữ cảnh & Bảng thuật ngữ", open=False):
                            sub_context_note_tb = gr.Textbox(
                                label="Ghi chú ngữ cảnh (Xưng hô nhân vật, chủ đề phim, mối quan hệ)",
                                placeholder="Ví dụ: Hai nhân vật là bạn thân xưng mày/tao; bối cảnh công nghệ thông tin...",
                                lines=2,
                            )
                            sub_glossary_tb = gr.Textbox(
                                label="Bảng thuật ngữ / Tên riêng (Mỗi dòng một mục: 'từ gốc: bản dịch')",
                                placeholder="Ví dụ:\nJohn: Tuấn\nDeep Learning: Học sâu",
                                lines=2,
                            )

                        with gr.Accordion("🎨 4. Kiểu dáng phụ đề & Ghép cứng (Burn-in)", open=False):
                            with gr.Row():
                                sub_burn_cb = gr.Checkbox(
                                    value=True,
                                    label="Ghép cứng phụ đề vào video (Burn-in MP4)",
                                )
                                sub_show_orig_cb = gr.Checkbox(
                                    value=False,
                                    label="Hiển thị song ngữ (Gốc ở trên, dịch ở dưới)",
                                )
                            sub_font_name_tb = gr.Textbox(
                                value="Noto Sans",
                                label="Phông chữ hiển thị (Hỗ trợ tiếng Việt có dấu, ví dụ: Noto Sans, Arial)",
                            )

                        with gr.Accordion("✍️ 5. Xem trước & Tùy chỉnh phụ đề gốc (Giải pháp can thiệp 2 bước)", open=True):
                            gr.Markdown(
                                "*Bạn có thể trích xuất bản nháp phụ đề để kiểm tra hoặc tự tay chỉnh sửa lỗi trước khi gửi sang Gemini dịch và ghép vào video.*"
                            )
                            with gr.Row():
                                btn_transcribe_preview = gr.Button("🔍 1. Chỉ nhận diện âm thanh (Tạo bản nháp SRT)", variant="secondary")
                            sub_preview_srt_tb = gr.Textbox(
                                label="Bản nháp phụ đề gốc (Có thể sửa trực tiếp hoặc dán phụ đề SRT có sẵn vào đây)",
                                placeholder="Bản nháp SRT sẽ xuất hiện ở đây sau khi bấm 'Chỉ nhận diện âm thanh'. Bạn có thể sửa câu từ, thời gian hoặc dán file SRT ngoài vào...",
                                lines=8,
                            )
                            with gr.Row():
                                btn_sub_from_preview = gr.Button("⚡ 2. Dịch & Ghép phụ đề từ bản nháp trên", variant="primary")

                    # ================= RIGHT COLUMN: CONTROLS & SUBTITLE OUTPUT =================
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("### 5. Điều khiển thực thi")
                            with gr.Row():
                                btn_sub_run = gr.Button("🚀 Bắt đầu tạo Phụ đề", variant="primary", size="lg", scale=4)
                                btn_sub_cancel = gr.Button("⏹️ Hủy", variant="stop", size="lg", scale=1)

                            sub_status_tb = gr.Textbox(
                                label="Trạng thái xử lý phụ đề",
                                value="Sẵn sàng",
                                interactive=False,
                                lines=2,
                            )

                        with gr.Group():
                            gr.Markdown("### 6. Tệp kết quả Phụ đề")
                            sub_out_video = gr.File(label="1. Video đã ghép phụ đề hoàn chỉnh (.mp4)")
                            with gr.Row():
                                sub_out_trans_srt = gr.File(label="2. Phụ đề tiếng Việt (.srt)")
                                sub_out_orig_srt = gr.File(label="3. Phụ đề gốc (.srt)")
                            with gr.Row():
                                sub_out_ass = gr.File(label="4. Phụ đề định dạng ASS (.ass)")
                                sub_out_json = gr.File(label="5. Báo cáo & Thống kê (.json)")

                        with gr.Accordion("💡 Hướng dẫn & Cơ chế giải phóng VRAM", open=False):
                            gr.Markdown(
                                """
                                - **Quy trình 1 chạm**: Tách âm thanh $\\rightarrow$ Nhận diện giọng nói đa ngôn ngữ (Whisper) $\\rightarrow$ Dịch tiếng Việt (Gemini) $\\rightarrow$ Xuất SRT/ASS $\\rightarrow$ Ghép cứng video (FFmpeg libass).
                                - **An toàn bộ nhớ GPU**: Mô hình Whisper tự động giải phóng 100% VRAM ngay sau khi nhận diện xong. Không xung đột bộ nhớ với các mô hình Upscale.
                                - **Bảo vệ nội dung**: Nếu video gốc đã là tiếng Việt, hệ thống tự động nhận diện và bỏ qua bước gọi API Gemini.
                                - **Tự động phục hồi**: Bộ nhớ đệm (cache) tự động lưu trên máy chủ, khi chạy lại các đoạn đã dịch sẽ không tốn thêm request API.
                                """
                            )

                sub_inputs = [
                    sub_in_file,
                    sub_source_lang_dd,
                    sub_target_lang_dd,
                    sub_audio_mode_dd,
                    sub_whisper_dd,
                    sub_gemini_dd,
                    sub_api_key_tb,
                    sub_context_note_tb,
                    sub_glossary_tb,
                    sub_burn_cb,
                    sub_show_orig_cb,
                    sub_font_name_tb,
                    sub_initial_prompt_tb,
                    sub_preview_srt_tb,
                    sub_use_demucs_cb,
                ]

                sub_outputs = [
                    sub_out_video,
                    sub_out_trans_srt,
                    sub_out_orig_srt,
                    sub_out_ass,
                    sub_out_json,
                    sub_status_tb,
                ]

                sub_run_event = btn_sub_run.click(
                    fn=run_subtitle,
                    inputs=sub_inputs,
                    outputs=sub_outputs,
                )

                sub_preview_event = btn_transcribe_preview.click(
                    fn=run_transcribe_only,
                    inputs=[
                        sub_in_file,
                        sub_source_lang_dd,
                        sub_audio_mode_dd,
                        sub_whisper_dd,
                        sub_initial_prompt_tb,
                        sub_use_demucs_cb,
                    ],
                    outputs=[sub_preview_srt_tb, sub_status_tb],
                )

                sub_from_preview_event = btn_sub_from_preview.click(
                    fn=run_subtitle,
                    inputs=sub_inputs,
                    outputs=sub_outputs,
                )

                btn_sub_cancel.click(
                    fn=cancel_subtitle,
                    outputs=[sub_status_tb],
                    cancels=[sub_run_event, sub_preview_event, sub_from_preview_event],
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Eco Video Upscaler - Gradio Cloud WebUI")
    parser.add_argument("--share", action="store_true", help="Generate public gradio.live link")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    parser.add_argument("--gemini-key", type=str, default="", help="Gemini API Key")
    args = parser.parse_args()

    if args.gemini_key and args.gemini_key.strip():
        k = args.gemini_key.strip()
        os.environ["GEMINI_API_KEY"] = k
        try:
            Path("/content/.gemini_key").write_text(k, encoding="utf-8")
        except Exception:
            pass

    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
