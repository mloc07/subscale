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
active_dubbing_pipeline = None
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


def cancel_dubbing():
    """Cancel currently running dubbing job."""
    global active_dubbing_pipeline
    if active_dubbing_pipeline is not None:
        active_dubbing_pipeline.cancel()
        return "⏹️ Đã gửi lệnh hủy tiến trình lồng tiếng."
    return "Không có tiến trình lồng tiếng nào đang chạy."


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


def run_dubbing(
    input_file_path: Optional[str],
    tts_backend_choice: str,
    whisper_model_choice: str,
    gemini_model_choice: str,
    num_speakers_choice: str,
    context_note_input: str,
    api_key_input: str = "",
    progress=gr.Progress(track_tqdm=False),
):
    """Execute AI Multi-Speaker Dubbing pipeline (S0 -> S9)."""
    global active_dubbing_pipeline
    if not input_file_path or not os.path.exists(input_file_path):
        raise gr.Error("Vui lòng chọn video đầu vào hợp lệ.")

    resolved_key = (api_key_input or "").strip()
    if not resolved_key:
        resolved_key = os.environ.get("GEMINI_API_KEY", "")
    if not resolved_key and os.path.exists("/content/.gemini_key"):
        try:
            resolved_key = Path("/content/.gemini_key").read_text(encoding="utf-8").strip()
        except Exception:
            pass

    if resolved_key:
        os.environ["GEMINI_API_KEY"] = resolved_key

    from ecoup.dubbing.config import DubbingConfig
    from ecoup.dubbing.pipeline import DubbingPipeline

    in_p = Path(input_file_path)
    out_dir = in_p.parent / f"dub_project_{in_p.stem}"

    cfg = DubbingConfig(project_dir=str(out_dir))
    tts_lower = (tts_backend_choice or "").lower()
    if "openvoice" in tts_lower or "clone âm sắc" in tts_lower:
        cfg.tts.backend = "openvoice"
    elif "google" in tts_lower or "gtts" in tts_lower:
        cfg.tts.backend = "gtts"
    elif "f5" in tts_lower:
        cfg.tts.backend = "f5"
    else:
        cfg.tts.backend = "mock"
    cfg.asr.model_size = whisper_model_choice
    cfg.translation.model = gemini_model_choice

    if num_speakers_choice and num_speakers_choice.isdigit():
        cfg.diarization.num_speakers = int(num_speakers_choice)
    else:
        cfg.diarization.num_speakers = None

    step_map = {
        "S0": ("Tiếp nhận & kiểm tra luồng âm thanh...", 0.05),
        "S1": ("Tách lời thoại & nhạc nền (Demucs 44.1kHz)...", 0.20),
        "S2": ("Nhận dạng giọng nói (Faster-Whisper)...", 0.40),
        "S3": ("Nhận diện & phân cụm nhân vật (SpeechBrain)...", 0.55),
        "S4": ("Trích xuất mẫu giọng sạch từng nhân vật...", 0.65),
        "S5": ("Dịch thuật có ngân sách thời lượng (Gemini)...", 0.75),
        "S6": ("Tổng hợp giọng tiếng Việt theo nhân vật...", 0.85),
        "S7": ("Khớp thời lượng từng câu thoại...", 0.92),
        "S8": ("Trộn âm thanh thoại & Ducking nhạc nền...", 0.96),
        "S9": ("Ghép âm thanh vào video (Stream copy)...", 0.99),
    }

    def on_dub_progress(step: str, frac: float):
        desc, base_pct = step_map.get(step, (f"Đang xử lý {step}...", 0.5))
        progress(base_pct, desc=desc)

    pipe = DubbingPipeline(input_file_path, config=cfg, on_progress=on_dub_progress)
    active_dubbing_pipeline = pipe

    try:
        progress(0.02, desc="Khởi động pipeline lồng tiếng...")
        res = pipe.run_full_pipeline(context_notes=context_note_input)
    except Exception as e:
        if pipe._cancelled:
            return None, None, None, None, "⏹️ Tiến trình lồng tiếng đã được hủy thành công."
        raise gr.Error(f"Lỗi lồng tiếng: {e}") from e
    finally:
        active_dubbing_pipeline = None

    final_vid = res.get("video_path")
    final_srt = res.get("srt_path")

    preview_audio = None
    spk0_prev = out_dir / "voices" / "spk_0" / "preview.wav"
    if spk0_prev.exists():
        preview_audio = str(spk0_prev)

    status_msg = f"🎉 Hoàn tất lồng tiếng AI! Tổng thời gian: {res.get('total_elapsed_s', 0)}s."
    return final_vid, final_vid, final_srt, preview_audio, status_msg


MAX_VC_SLOTS = 8
active_vc_dub_pipeline = None


def _resolve_gemini_api_key(api_key_input: str = "") -> str:
    resolved_key = (api_key_input or "").strip()
    if not resolved_key:
        resolved_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not resolved_key and os.path.exists("/content/.gemini_key"):
        try:
            resolved_key = Path("/content/.gemini_key").read_text(encoding="utf-8").strip()
        except Exception:
            resolved_key = ""
    if resolved_key:
        os.environ["GEMINI_API_KEY"] = resolved_key
    return resolved_key


def _vc_pitch_mode_value(choice: str) -> str:
    low = (choice or "").lower()
    if "giữ nguyên" in low or "none" in low:
        return "none"
    if "thủ công" in low or "chỉnh tay" in low or "manual" in low:
        return "manual"
    return "auto"


def _vc_target_special_options() -> list[str]:
    return [
        "Tự clone từ chính nhân vật trong video [__self__]",
        "Giữ nguyên thoại gốc [__original__]",
    ]


def _vc_build_bank_ui_updates(bank, status_msg: str, entries_override=None):
    """Build table, dropdown, and 8 dynamic slot updates for Sub-tab 4.1 & 4.3."""
    all_entries = bank.list_voices()
    shown = entries_override if entries_override is not None else all_entries

    rows = [
        [
            e.clone_id,
            e.display_name,
            e.speaker_id,
            f"{e.speaking_duration_s:.1f}s",
            f"{e.estimated_f0_hz:.0f} Hz" if e.estimated_f0_hz > 0 else "N/A",
            f"{e.quality_score:.2f}",
            (e.reference_text or "")[:75],
        ]
        for e in shown
    ]

    choices = bank.get_dropdown_choices()
    dd_update = gr.update(choices=choices, value=choices[0] if choices else None)

    slot_updates = []
    for idx in range(MAX_VC_SLOTS):
        if idx < len(shown):
            e = shown[idx]
            f0_str = f"~{e.estimated_f0_hz:.0f} Hz" if e.estimated_f0_hz > 0 else "F0 N/A"
            prev_path = e.preview_mp3_path if (e.preview_mp3_path and os.path.exists(e.preview_mp3_path)) else None
            ref_path = e.ref_wav_path if (e.ref_wav_path and os.path.exists(e.ref_wav_path)) else None
            slot_updates.extend([
                gr.update(visible=True),
                gr.update(value=e.clone_id),
                gr.update(value=e.display_name),
                gr.update(value=prev_path, label=f"Clone {idx + 1}: {e.display_name} — Bản tiếng Việt (.mp3)"),
                gr.update(value=ref_path, label=f"Giọng gốc sạch ({e.speaker_id} | {e.speaking_duration_s:.1f}s | {f0_str})"),
            ])
        else:
            slot_updates.extend([
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=None),
                gr.update(value=None),
            ])

    return (rows, status_msg, dd_update, *slot_updates)


def vc_run_clone_voices(
    source_file_path: Optional[str],
    label_prefix: str,
    num_speakers_choice: str,
    distance_threshold: float,
    whisper_model: str,
    pitch_mode_choice: str,
    progress=gr.Progress(track_tqdm=False),
):
    """Sub-tab 4.1: Extract and clone all detected speakers from source video/audio."""
    if not source_file_path or not os.path.exists(source_file_path):
        raise gr.Error("Vui lòng tải lên tệp Video/Audio nguồn hợp lệ.")

    from ecoup.dubbing.voice_bank import VoiceBankManager, clone_voices_from_video

    num_spk = int(num_speakers_choice) if (num_speakers_choice and num_speakers_choice.isdigit()) else None
    pitch_mode = _vc_pitch_mode_value(pitch_mode_choice)

    step_pct = {
        "S0": (0.05, "Kiểm tra âm thanh nguồn..."),
        "S1": (0.25, "Tách lời thoại & lọc tạp âm (Demucs)..."),
        "S2": (0.45, "Nhận dạng lời thoại (Whisper ASR)..."),
        "S3": (0.62, "Phân cụm từng nhân vật (SpeechBrain)..."),
        "S4": (0.75, "Trích xuất đoạn mẫu giọng sạch từng nhân vật..."),
        "S4B_CLONE": (0.90, "Nhân bản embedding giọng & tạo mẫu nghe thử .mp3..."),
    }

    def _on_prog(step: str, frac: float):
        base, desc = step_pct.get(step, (0.5, f"Đang xử lý {step}..."))
        progress(min(0.99, base + frac * 0.08), desc=desc)

    progress(0.02, desc="Bắt đầu phân tích & clone giọng...")
    entries = clone_voices_from_video(
        video_path=source_file_path,
        label_prefix=label_prefix,
        num_speakers=num_spk,
        distance_threshold=float(distance_threshold),
        whisper_model=whisper_model,
        pitch_match_mode=pitch_mode,
        on_progress=_on_prog,
    )

    bank = VoiceBankManager()
    names = ", ".join(e.display_name for e in entries)
    status = f"✅ Đã clone thành công {len(entries)} giọng nói từ '{Path(source_file_path).name}': {names}."
    return _vc_build_bank_ui_updates(bank, status, entries_override=bank.list_voices())


def vc_refresh_bank():
    from ecoup.dubbing.voice_bank import VoiceBankManager

    bank = VoiceBankManager()
    voices = bank.list_voices()
    status = f"📚 Kho Giọng Clone hiện có {len(voices)} bản lưu."
    return _vc_build_bank_ui_updates(bank, status, entries_override=voices)


def vc_rename_clone_slot(clone_id: str, new_name: str):
    from ecoup.dubbing.voice_bank import VoiceBankManager

    bank = VoiceBankManager()
    cid = (clone_id or "").strip()
    if not cid:
        return _vc_build_bank_ui_updates(bank, "⚠️ Không xác định được Clone ID.")
    updated = bank.rename_voice(cid, new_name)
    return _vc_build_bank_ui_updates(bank, f"✏️ Đã đổi tên thành '{updated.display_name}'.")


def vc_delete_clone_slot(clone_id: str):
    from ecoup.dubbing.voice_bank import VoiceBankManager

    bank = VoiceBankManager()
    cid = (clone_id or "").strip()
    ok = bank.delete_voice(cid) if cid else False
    msg = f"🗑️ Đã xóa bản clone '{cid}'." if ok else f"⚠️ Không tìm thấy '{cid}'."
    return _vc_build_bank_ui_updates(bank, msg)


def _vc_build_target_slot_updates(bank, speakers: list[dict], mapping: dict[str, str]):
    choices = bank.get_dropdown_choices() + _vc_target_special_options()
    label_by_id = {v.clone_id: bank.format_choice_label(v) for v in bank.list_voices()}
    label_by_id["__self__"] = _vc_target_special_options()[0]
    label_by_id["__original__"] = _vc_target_special_options()[1]

    slot_updates = []
    for idx in range(MAX_VC_SLOTS):
        if idx < len(speakers):
            spk = speakers[idx]
            spk_id = spk["speaker_id"]
            f0_txt = f"~{spk['estimated_f0_hz']:.0f} Hz" if spk.get("estimated_f0_hz", 0) > 0 else "N/A"
            prev_wav = spk.get("preview_wav") if (spk.get("preview_wav") and os.path.exists(spk["preview_wav"])) else None
            txt = spk.get("sample_text") or "(Không có transcript)"
            md = (
                f"**{spk['label']}** — Thời lượng nói: **{spk['speaking_duration_s']:.1f}s** ({spk['total_segments']} câu) | "
                f"Cao độ: **{f0_txt}**  \n*Thoại mẫu:* \"{txt}\""
            )
            chosen_id = mapping.get(spk_id, "__self__")
            chosen_label = label_by_id.get(chosen_id, choices[0] if choices else None)
            slot_updates.extend([
                gr.update(visible=True),
                gr.update(value=md),
                gr.update(value=prev_wav, label=f"Nghe thử {spk['label']}"),
                gr.update(choices=choices, value=chosen_label),
            ])
        else:
            slot_updates.extend([
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=None),
                gr.update(choices=choices, value=None),
            ])
    return slot_updates


def vc_analyze_target_video(
    target_video_path: Optional[str],
    num_speakers_choice: str,
    whisper_model: str,
    progress=gr.Progress(track_tqdm=False),
):
    """Sub-tab 4.2 Step A: Analyze speakers in target Video 2 and auto-map order to clones."""
    if not target_video_path or not os.path.exists(target_video_path):
        raise gr.Error("Vui lòng tải lên Video đích cần lồng tiếng.")

    from ecoup.dubbing.config import DubbingConfig
    from ecoup.dubbing.pipeline import DubbingPipeline
    from ecoup.dubbing.voice_bank import VoiceBankManager, suggest_speaker_mapping

    in_p = Path(target_video_path)
    proj_dir = in_p.parent / f"dub_project_{in_p.stem}"

    cfg = DubbingConfig(project_dir=str(proj_dir))
    cfg.asr.model_size = whisper_model
    cfg.diarization.num_speakers = (
        int(num_speakers_choice) if (num_speakers_choice and num_speakers_choice.isdigit()) else None
    )

    def _on_prog(step: str, frac: float):
        progress(min(0.98, 0.1 + frac * 0.2), desc=f"Phân tích video đích ({step}: {int(frac * 100)}%)...")

    progress(0.05, desc="Khởi chạy phân tách & nhận diện nhân vật trong Video đích...")
    pipe = DubbingPipeline(target_video_path, config=cfg, on_progress=_on_prog)
    speakers = pipe.analyze_target_speakers()

    bank = VoiceBankManager()
    bank_voices = bank.list_voices()
    default_map = suggest_speaker_mapping(speakers, bank_voices, mode="order")

    status = (
        f"🔍 Đã phát hiện {len(speakers)} nhân vật trong '{in_p.name}'. "
        + (f"Đã tự động gán theo thứ tự với {len(bank_voices)} bản Clone." if bank_voices else "Kho Clone hiện trống — mặc định tự clone từ video gốc.")
    )
    slot_updates = _vc_build_target_slot_updates(bank, speakers, default_map)
    return (speakers, status, *slot_updates)


def vc_suggest_mapping_f0(speakers_state: list[dict]):
    from ecoup.dubbing.voice_bank import VoiceBankManager, suggest_speaker_mapping

    bank = VoiceBankManager()
    if not speakers_state:
        return ("⚠️ Hãy phân tích Video đích trước khi gợi ý ghép.", *[gr.update() for _ in range(MAX_VC_SLOTS)])
    voices = bank.list_voices()
    f0_map = suggest_speaker_mapping(speakers_state, voices, mode="f0")

    choices = bank.get_dropdown_choices() + _vc_target_special_options()
    label_by_id = {v.clone_id: bank.format_choice_label(v) for v in voices}
    label_by_id["__self__"] = _vc_target_special_options()[0]
    label_by_id["__original__"] = _vc_target_special_options()[1]

    dd_updates = []
    for idx in range(MAX_VC_SLOTS):
        if idx < len(speakers_state):
            spk_id = speakers_state[idx]["speaker_id"]
            chosen_id = f0_map.get(spk_id, "__self__")
            dd_updates.append(gr.update(choices=choices, value=label_by_id.get(chosen_id, choices[0])))
        else:
            dd_updates.append(gr.update())
    return ("🎯 Đã gợi ý ghép theo cao độ F0 (Nam/Nữ gần nhất).", *dd_updates)


def vc_refresh_target_dropdowns(speakers_state: list[dict], *current_dd_vals):
    from ecoup.dubbing.voice_bank import VoiceBankManager, suggest_speaker_mapping

    bank = VoiceBankManager()
    voices = bank.list_voices()
    choices = bank.get_dropdown_choices() + _vc_target_special_options()
    default_map = suggest_speaker_mapping(speakers_state or [], voices, mode="order")
    label_by_id = {v.clone_id: bank.format_choice_label(v) for v in voices}
    label_by_id["__self__"] = _vc_target_special_options()[0]
    label_by_id["__original__"] = _vc_target_special_options()[1]

    dd_updates = []
    for idx in range(MAX_VC_SLOTS):
        cur = current_dd_vals[idx] if idx < len(current_dd_vals) else None
        cur_id = VoiceBankManager._normalize_choice_id(cur or "")
        if cur_id in label_by_id:
            dd_updates.append(gr.update(choices=choices, value=label_by_id[cur_id]))
        elif speakers_state and idx < len(speakers_state):
            fallback_id = default_map.get(speakers_state[idx]["speaker_id"], "__self__")
            dd_updates.append(gr.update(choices=choices, value=label_by_id.get(fallback_id, choices[0])))
        else:
            dd_updates.append(gr.update(choices=choices, value=None))
    return tuple(dd_updates)


def vc_run_dub_with_mapping(
    target_video_path: Optional[str],
    speakers_state: list[dict],
    whisper_model: str,
    num_speakers_choice: str,
    gemini_model: str,
    context_notes: str,
    api_key_input: str,
    pitch_mode_choice: str,
    pitch_shift_val: float,
    *slot_dd_vals,
    progress=gr.Progress(track_tqdm=False),
):
    """Sub-tab 4.2 Step C: Dub target Video 2 using the configured speaker->clone mapping."""
    global active_vc_dub_pipeline
    if not target_video_path or not os.path.exists(target_video_path):
        raise gr.Error("Vui lòng tải lên Video đích cần lồng tiếng.")

    _resolve_gemini_api_key(api_key_input)

    from ecoup.dubbing.config import DubbingConfig
    from ecoup.dubbing.pipeline import DubbingPipeline
    from ecoup.dubbing.voice_bank import VoiceBankManager

    bank = VoiceBankManager()
    spk_map: dict[str, str] = {}
    for idx, spk in enumerate(speakers_state or []):
        if idx >= MAX_VC_SLOTS:
            break
        raw_choice = slot_dd_vals[idx] if idx < len(slot_dd_vals) else ""
        norm_id = VoiceBankManager._normalize_choice_id(raw_choice or "")
        if norm_id:
            spk_map[str(spk["speaker_id"])] = norm_id

    in_p = Path(target_video_path)
    proj_dir = in_p.parent / f"dub_project_{in_p.stem}"

    cfg = DubbingConfig(project_dir=str(proj_dir))
    cfg.asr.model_size = whisper_model
    cfg.diarization.num_speakers = (
        int(num_speakers_choice) if (num_speakers_choice and num_speakers_choice.isdigit()) else None
    )
    cfg.translation.model = gemini_model
    cfg.tts.backend = "openvoice"
    cfg.tts.voice_bank_dir = str(bank.bank_dir)
    cfg.tts.speaker_voice_map = spk_map
    cfg.tts.pitch_match_mode = _vc_pitch_mode_value(pitch_mode_choice)
    cfg.tts.pitch_shift_semitones = float(pitch_shift_val or 0.0)

    step_map = {
        "S0": (0.05, "Kiểm tra âm thanh video đích..."),
        "S1": (0.15, "Tách thoại & nhạc nền (Demucs)..."),
        "S2": (0.30, "Nhận dạng giọng nói (Whisper)..."),
        "S3": (0.42, "Phân vai nhân vật (SpeechBrain)..."),
        "S4": (0.50, "Chuẩn bị hồ sơ giọng..."),
        "S5": (0.62, "Dịch lời thoại khớp thời lượng (Gemini)..."),
        "S6": (0.78, "Nhân bản giọng từ Kho Clone (OpenVoice v2)..."),
        "S7": (0.88, "Khớp nhịp & tốc độ từng câu..."),
        "S8": (0.94, "Trộn nhạc nền & ducking thoại..."),
        "S9": (0.98, "Ghép luồng video gốc (-c:v copy)..."),
    }

    def _on_prog(step: str, frac: float):
        base, desc = step_map.get(step, (0.5, f"Đang xử lý {step}..."))
        progress(min(0.99, base + frac * 0.04), desc=desc)

    pipe = DubbingPipeline(target_video_path, config=cfg, on_progress=_on_prog)
    active_vc_dub_pipeline = pipe
    try:
        progress(0.02, desc="Đang chạy pipeline lồng tiếng theo giọng clone...")
        res = pipe.run_full_pipeline(context_notes=context_notes or "")
    except Exception as e:
        if pipe._cancelled:
            return None, None, None, "⏹️ Tiến trình lồng tiếng đã được hủy."
        raise gr.Error(f"Lỗi lồng tiếng theo giọng clone: {e}") from e
    finally:
        active_vc_dub_pipeline = None

    out_vid = res.get("video_path")
    out_srt = res.get("srt_path")
    status = f"🎉 Hoàn tất lồng tiếng Video bằng giọng Clone! Tổng thời gian: {res.get('total_elapsed_s', 0)}s."
    return out_vid, out_vid, out_srt, status


def vc_cancel_dub():
    global active_vc_dub_pipeline
    if active_vc_dub_pipeline is not None:
        active_vc_dub_pipeline.cancel()
        return "⏹️ Đang gửi tín hiệu dừng lồng tiếng..."
    return "ℹ️ Hiện không có tiến trình lồng tiếng Studio nào đang chạy."


def vc_preview_clone_for_tts(clone_choice: Optional[str]):
    from ecoup.dubbing.voice_bank import VoiceBankManager

    if not clone_choice:
        return None, "Chưa chọn bản giọng clone."
    bank = VoiceBankManager()
    entry = bank.get_voice(clone_choice)
    if entry is None:
        return None, "Không tìm thấy thông tin bản clone trong kho."
    audio_path = entry.preview_mp3_path if (entry.preview_mp3_path and os.path.exists(entry.preview_mp3_path)) else entry.ref_wav_path
    f0_txt = f"{entry.estimated_f0_hz:.0f} Hz" if entry.estimated_f0_hz > 0 else "N/A"
    info = (
        f"**{entry.display_name}** (`{entry.clone_id}`) | Nguồn: `{entry.source_video}` ({entry.speaker_id}) | "
        f"Thời lượng mẫu: `{entry.speaking_duration_s:.1f}s` | Cao độ F0: `{f0_txt}`"
    )
    return (audio_path if audio_path and os.path.exists(audio_path) else None), info


def vc_run_tts_export_mp3(
    clone_choice: Optional[str],
    text_input: str,
    lang_choice: str,
    speed: float,
    pitch_mode_choice: str,
    pitch_shift_val: float,
    progress=gr.Progress(track_tqdm=False),
):
    """Sub-tab 4.3: Read text with selected cloned voice and export MP3."""
    if not clone_choice:
        raise gr.Error("Vui lòng chọn một bản giọng clone (hãy clone giọng ở Tab 4.1 trước).")
    if not (text_input or "").strip():
        raise gr.Error("Vui lòng nhập nội dung văn bản cần đọc.")

    from ecoup.dubbing.tts import OpenVoiceBackend, synthesize_text_to_mp3
    from ecoup.dubbing.voice_bank import VoiceBankManager

    bank = VoiceBankManager()
    ref_path = bank.resolve_voice(clone_choice)
    if ref_path is None or not ref_path.exists():
        raise gr.Error(f"Không tìm thấy tệp mẫu của giọng '{clone_choice}'.")

    clone_id = VoiceBankManager._normalize_choice_id(clone_choice)
    lang = "en" if "(en)" in (lang_choice or "").lower() else "vi"
    pitch_mode = _vc_pitch_mode_value(pitch_mode_choice)

    out_dir = bank.bank_dir / "_tts_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    out_mp3 = out_dir / f"tts_{clone_id}_{ts_str}.mp3"

    backend = OpenVoiceBackend(
        device="auto",
        base_lang=lang,
        pitch_match_mode=pitch_mode,
        pitch_shift_semitones=float(pitch_shift_val or 0.0),
    )

    def _on_prog(frac: float):
        progress(min(0.98, 0.1 + frac * 0.85), desc=f"Đang tổng hợp đoạn văn bản ({int(frac * 100)}%)...")

    try:
        progress(0.05, desc="Đang nạp mô hình OpenVoice v2 & tổng hợp giọng...")
        rep = synthesize_text_to_mp3(
            text=text_input,
            output_mp3=out_mp3,
            backend=backend,
            ref_audio_path=ref_path,
            speed=float(speed or 1.0),
            speaker_id=clone_id,
            bitrate="192k",
            on_progress=_on_prog,
        )
    finally:
        if getattr(backend, "_converter", None) is not None:
            try:
                backend._converter = None
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    status = (
        f"✅ Đã đọc xong {rep['chunks']} đoạn câu ({rep['duration_s']}s) bằng giọng '{clone_id}' "
        f"-> `{Path(rep['mp3_path']).name}`."
    )
    return rep["mp3_path"], rep["mp3_path"], status


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

            with gr.Tab("🎙️ Lồng tiếng AI (Multi-Speaker Dubbing)"):
                with gr.Row():
                    # ================= LEFT COLUMN: INPUTS =================
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("### 1. Tệp video nguồn")
                            dub_in_file = gr.File(
                                label="Kéo thả hoặc chọn video (MP4, MKV, MOV, WebM)",
                                file_types=["video"],
                                type="filepath",
                            )

                        with gr.Group():
                            gr.Markdown("### 2. Cấu hình Giọng nói & Nhận dạng")
                            with gr.Row():
                                dub_tts_backend_dd = gr.Dropdown(
                                    label="Bộ sinh giọng tiếng Việt (TTS)",
                                    choices=[
                                        "OpenVoice v2 (Clone âm sắc giọng gốc + F0)",
                                        "Google Cloud / gTTS (Phát âm chuẩn 100%)",
                                        "F5-TTS (Zero-shot Voice Clone AI)",
                                    ],
                                    value="OpenVoice v2 (Clone âm sắc giọng gốc + F0)",
                                )
                                dub_whisper_dd = gr.Dropdown(
                                    label="Mô hình Whisper ASR",
                                    choices=["base", "small", "medium", "large-v3"],
                                    value="base",
                                )
                            with gr.Row():
                                dub_num_spk_dd = gr.Dropdown(
                                    label="Số nhân vật",
                                    choices=["Tự động phát hiện (Auto)", "1", "2", "3", "4", "5"],
                                    value="Tự động phát hiện (Auto)",
                                )
                                dub_gemini_dd = gr.Dropdown(
                                    label="Mô hình Dịch thuật",
                                    choices=["gemini-2.5-flash", "gemini-1.5-flash"],
                                    value="gemini-2.5-flash",
                                )

                        with gr.Group():
                            gr.Markdown("### 3. Ngữ cảnh & API Key")
                            dub_context_tb = gr.Textbox(
                                label="Ghi chú ngữ cảnh / Bối cảnh phim (tùy chọn)",
                                placeholder="Ví dụ: Phim hoạt hình hài hước, xưng hô anh/em, văn phong tự nhiên...",
                                lines=2,
                            )
                            dub_api_key_tb = gr.Textbox(
                                label="Google Gemini API Key (tự động đọc nếu đã nạp)",
                                placeholder="Nhập API Key nếu chưa lưu trên máy...",
                                type="password",
                            )

                        with gr.Row():
                            btn_dub_run = gr.Button("🚀 Bắt đầu Lồng tiếng (1-Click)", variant="primary", scale=2)
                            btn_dub_cancel = gr.Button("⏹️ Hủy tiến trình", variant="stop", scale=1)

                    # ================= RIGHT COLUMN: OUTPUTS =================
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("### 🎬 Video Lồng tiếng Thành phẩm")
                            dub_out_video = gr.Video(label="Xem trước video lồng tiếng", interactive=False)
                            dub_status_tb = gr.Textbox(label="Trạng thái & Tiến độ", interactive=False, lines=2)

                        with gr.Group():
                            gr.Markdown("### 📥 Tải kết quả về máy")
                            with gr.Row():
                                dub_dl_video = gr.File(label="Tải Video (.mp4)")
                                dub_dl_srt = gr.File(label="Tải Phụ đề (.srt)")
                            dub_preview_audio = gr.Audio(label="Mẫu giọng nhân vật chính (Clean Reference Voice)", interactive=False)

                dub_inputs = [
                    dub_in_file,
                    dub_tts_backend_dd,
                    dub_whisper_dd,
                    dub_gemini_dd,
                    dub_num_spk_dd,
                    dub_context_tb,
                    dub_api_key_tb,
                ]

                dub_outputs = [
                    dub_out_video,
                    dub_dl_video,
                    dub_dl_srt,
                    dub_preview_audio,
                    dub_status_tb,
                ]

                dub_run_event = btn_dub_run.click(
                    fn=run_dubbing,
                    inputs=dub_inputs,
                    outputs=dub_outputs,
                )

                btn_dub_cancel.click(
                    fn=cancel_dubbing,
                    outputs=[dub_status_tb],
                    cancels=[dub_run_event],
                )

            with gr.Tab("🧬 Voice Cloning Studio (Clone đa giọng & TTS MP3)"):
                gr.Markdown(
                    "Nhân bản một hoặc nhiều giọng nói từ Video/Audio vào **Kho Giọng Clone**, "
                    "sau đó dùng lại cho Video khác theo từng nhân vật hoặc đọc văn bản xuất `.mp3`."
                )

                with gr.Tabs():
                    # -------- Sub-tab 4.1: Clone voices from Video/Audio --------
                    with gr.Tab("1. Clone giọng từ Video / Audio"):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Group():
                                    gr.Markdown("### Nguồn trích xuất giọng nói")
                                    vc_src_file = gr.File(
                                        label="Tải lên Video hoặc Audio nguồn (1 hoặc nhiều nhân vật)",
                                        file_types=["video", "audio"],
                                        type="filepath",
                                    )
                                    vc_src_label_tb = gr.Textbox(
                                        label="Tên gợi nhớ bộ giọng (Label)",
                                        placeholder="Ví dụ: FamilyGuy_S1, NhanVatA...",
                                    )
                                    with gr.Row():
                                        vc_src_num_spk_dd = gr.Dropdown(
                                            label="Số lượng giọng dự kiến",
                                            choices=["Tự động phát hiện (Auto)", "1", "2", "3", "4", "5", "6", "7", "8"],
                                            value="Tự động phát hiện (Auto)",
                                        )
                                        vc_src_whisper_dd = gr.Dropdown(
                                            label="Whisper ASR",
                                            choices=["base", "small", "medium", "large-v3"],
                                            value="medium",
                                        )
                                    with gr.Row():
                                        vc_src_thresh_sl = gr.Slider(
                                            minimum=0.45,
                                            maximum=0.95,
                                            value=0.72,
                                            step=0.02,
                                            label="Ngưỡng tách người nói (nhỏ = tách nhạy hơn)",
                                        )
                                        vc_src_pitch_dd = gr.Dropdown(
                                            label="Khớp cao độ F0 bản nghe thử",
                                            choices=[
                                                "Tự động khớp cao độ F0 (Khuyên dùng)",
                                                "Giữ nguyên gốc",
                                            ],
                                            value="Tự động khớp cao độ F0 (Khuyên dùng)",
                                        )
                                with gr.Row():
                                    btn_vc_clone = gr.Button(
                                        "🧬 Phân tích & Xuất các bản Clone",
                                        variant="primary",
                                        size="lg",
                                    )
                                    btn_vc_refresh_bank = gr.Button(
                                        "🔄 Tải lại Kho Giọng",
                                        variant="secondary",
                                    )
                                vc_clone_status = gr.Textbox(
                                    label="Trạng thái Kho Giọng Clone",
                                    value="Sẵn sàng. Tải lên video có nhiều nhân vật để xuất đồng thời nhiều bản Clone.",
                                    interactive=False,
                                    lines=2,
                                )

                            with gr.Column(scale=7):
                                vc_clone_table = gr.Dataframe(
                                    headers=[
                                        "Clone ID",
                                        "Tên hiển thị",
                                        "Vai gốc",
                                        "Thời lượng",
                                        "Cao độ F0",
                                        "Chất lượng",
                                        "Thoại mẫu",
                                    ],
                                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                                    label="Danh sách Giọng Clone đã lưu trong Kho",
                                    interactive=False,
                                )

                        gr.Markdown("### Nghe thử từng bản Clone & Quản lý (tối đa 8 nhân vật)")
                        vc_slot_all_comps = []
                        vc_slot_actions = []
                        for _slot_i in range(MAX_VC_SLOTS):
                            with gr.Group(visible=False) as _sg:
                                _sid = gr.Textbox(label="Clone ID", interactive=False, visible=False)
                                with gr.Row():
                                    _sname = gr.Textbox(label=f"Tên bản Clone #{_slot_i + 1}", scale=4)
                                    _bren = gr.Button("💾 Lưu tên", scale=1)
                                    _bdel = gr.Button("🗑️ Xóa", variant="stop", scale=1)
                                with gr.Row():
                                    _sprev = gr.Audio(
                                        label=f"Clone {_slot_i + 1} — Nghe thử Tiếng Việt (.mp3)",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    _sref = gr.Audio(
                                        label=f"Giọng gốc sạch (.wav)",
                                        type="filepath",
                                        interactive=False,
                                    )
                            vc_slot_all_comps.extend([_sg, _sid, _sname, _sprev, _sref])
                            vc_slot_actions.append((_sid, _sname, _bren, _bdel))

                    # -------- Sub-tab 4.2: Cross-video Speaker Analysis & Dubbing --------
                    with gr.Tab("2. Lồng tiếng Video bằng Giọng Clone"):
                        vc_target_speakers_state = gr.State([])
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Group():
                                    gr.Markdown("### Bước A: Phân tích nhân vật Video đích")
                                    vc_tgt_video = gr.File(
                                        label="Tải lên Video thứ 2 cần lồng tiếng bằng giọng Clone",
                                        file_types=["video"],
                                        type="filepath",
                                    )
                                    with gr.Row():
                                        vc_tgt_num_spk_dd = gr.Dropdown(
                                            label="Số nhân vật Video đích",
                                            choices=["Tự động phát hiện (Auto)", "1", "2", "3", "4", "5", "6", "7", "8"],
                                            value="Tự động phát hiện (Auto)",
                                        )
                                        vc_tgt_whisper_dd = gr.Dropdown(
                                            label="Whisper ASR",
                                            choices=["base", "small", "medium", "large-v3"],
                                            value="medium",
                                        )
                                    btn_vc_analyze = gr.Button(
                                        "🔍 Phân tích nhân vật trong Video",
                                        variant="secondary",
                                    )
                                    vc_target_status = gr.Textbox(
                                        label="Kết quả phân tích nhân vật đích",
                                        value="Chưa phân tích Video đích.",
                                        interactive=False,
                                        lines=2,
                                    )

                                with gr.Group():
                                    gr.Markdown("### Bước C: Cấu hình dịch & Lồng tiếng")
                                    with gr.Row():
                                        vc_tgt_gemini_dd = gr.Dropdown(
                                            label="Mô hình Dịch thuật",
                                            choices=["gemini-2.5-flash", "gemini-1.5-flash"],
                                            value="gemini-2.5-flash",
                                        )
                                        vc_tgt_pitch_dd = gr.Dropdown(
                                            label="Khớp cao độ F0",
                                            choices=[
                                                "Tự động khớp cao độ F0 (Khuyên dùng)",
                                                "Giữ nguyên gốc",
                                                "Thủ công (semitones)",
                                            ],
                                            value="Tự động khớp cao độ F0 (Khuyên dùng)",
                                        )
                                    vc_tgt_pitch_shift_sl = gr.Slider(
                                        minimum=-7.0,
                                        maximum=7.0,
                                        value=0.0,
                                        step=0.5,
                                        label="Dịch cao độ thủ công (semitones — chỉ dùng khi chọn Thủ công)",
                                    )
                                    vc_tgt_context_tb = gr.Textbox(
                                        label="Ngữ cảnh & phong cách dịch (tùy chọn)",
                                        placeholder="Ví dụ: Phim hài gia đình, xưng hô tự nhiên...",
                                        lines=2,
                                    )
                                    vc_tgt_api_key_tb = gr.Textbox(
                                        label="Gemini API Key (nếu chưa cấu hình sẵn trên hệ thống)",
                                        placeholder="Để trống nếu đã nạp sẵn GEMINI_API_KEY...",
                                        type="password",
                                    )
                                    with gr.Row():
                                        btn_vc_dub_run = gr.Button(
                                            "🎬 Bắt đầu Lồng tiếng theo Giọng đã gán",
                                            variant="primary",
                                            size="lg",
                                            scale=3,
                                        )
                                        btn_vc_dub_cancel = gr.Button(
                                            "⏹️ Hủy",
                                            variant="stop",
                                            size="lg",
                                            scale=1,
                                        )

                            with gr.Column(scale=7):
                                with gr.Group():
                                    gr.Markdown("### Bước B: Ánh xạ từng nhân vật đích ↔ Bản Clone")
                                    with gr.Row():
                                        btn_vc_suggest_f0 = gr.Button("🎯 Gợi ý ghép theo Cao độ F0", size="sm")
                                        btn_vc_refresh_map = gr.Button("🔄 Làm mới danh sách Clone", size="sm")

                                    vc_tgt_slot_all_comps = []
                                    vc_tgt_dd_list = []
                                    for _t_i in range(MAX_VC_SLOTS):
                                        with gr.Group(visible=False) as _tg:
                                            _tmd = gr.Markdown(f"**Giọng {_t_i + 1}**")
                                            with gr.Row():
                                                _tprev = gr.Audio(
                                                    label=f"Nghe thử Giọng {_t_i + 1}",
                                                    type="filepath",
                                                    interactive=False,
                                                )
                                                _tdd = gr.Dropdown(
                                                    label=f"Ánh xạ Giọng {_t_i + 1} -> Bản Clone",
                                                    choices=_vc_target_special_options(),
                                                    value=_vc_target_special_options()[0],
                                                    allow_custom_value=True,
                                                )
                                        vc_tgt_slot_all_comps.extend([_tg, _tmd, _tprev, _tdd])
                                        vc_tgt_dd_list.append(_tdd)

                                with gr.Group():
                                    gr.Markdown("### Thành phẩm Video Lồng tiếng")
                                    vc_tgt_dub_status = gr.Textbox(
                                        label="Trạng thái Lồng tiếng Studio",
                                        value="Sẵn sàng.",
                                        interactive=False,
                                    )
                                    vc_tgt_out_video = gr.Video(
                                        label="Video Lồng tiếng hoàn chỉnh (MP4)",
                                        interactive=False,
                                    )
                                    with gr.Row():
                                        vc_tgt_dl_video = gr.File(label="Tải về Video (.mp4)")
                                        vc_tgt_dl_srt = gr.File(label="Tải về Phụ đề (.srt)")

                    # -------- Sub-tab 4.3: Select Cloned Voice -> Read TTS -> Export .MP3 --------
                    with gr.Tab("3. Đọc văn bản (TTS) & Xuất .MP3"):
                        with gr.Row():
                            with gr.Column(scale=6):
                                with gr.Group():
                                    gr.Markdown("### Chọn giọng Clone & Nhập văn bản")
                                    vc_tts_clone_dd = gr.Dropdown(
                                        label="Chọn bản Giọng Clone từ Kho",
                                        choices=[],
                                        value=None,
                                        allow_custom_value=True,
                                    )
                                    vc_tts_clone_info = gr.Markdown("Chưa chọn giọng clone.")
                                    vc_tts_ref_preview = gr.Audio(
                                        label="Nghe thử mẫu giọng Clone đang chọn",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    vc_tts_text_tb = gr.Textbox(
                                        label="Nội dung văn bản cần đọc (hỗ trợ nhiều câu dài)",
                                        placeholder="Nhập đoạn văn bản tiếng Việt hoặc tiếng Anh để giọng clone đọc...",
                                        lines=6,
                                    )
                                    with gr.Row():
                                        vc_tts_lang_dd = gr.Dropdown(
                                            label="Ngôn ngữ đọc",
                                            choices=["Tiếng Việt (vi)", "English (en)"],
                                            value="Tiếng Việt (vi)",
                                        )
                                        vc_tts_speed_sl = gr.Slider(
                                            minimum=0.65,
                                            maximum=1.55,
                                            value=1.0,
                                            step=0.05,
                                            label="Tốc độ đọc",
                                        )
                                    with gr.Row():
                                        vc_tts_pitch_dd = gr.Dropdown(
                                            label="Khớp cao độ F0",
                                            choices=[
                                                "Tự động khớp cao độ F0 (Khuyên dùng)",
                                                "Giữ nguyên gốc",
                                                "Thủ công (semitones)",
                                            ],
                                            value="Tự động khớp cao độ F0 (Khuyên dùng)",
                                        )
                                        vc_tts_pitch_shift_sl = gr.Slider(
                                            minimum=-7.0,
                                            maximum=7.0,
                                            value=0.0,
                                            step=0.5,
                                            label="Dịch cao độ thủ công (semitones)",
                                        )
                                    btn_vc_tts_run = gr.Button(
                                        "🔊 Đọc & Xuất file .MP3",
                                        variant="primary",
                                        size="lg",
                                    )

                            with gr.Column(scale=6):
                                with gr.Group():
                                    gr.Markdown("### Kết quả Âm thanh .MP3")
                                    vc_tts_status = gr.Textbox(
                                        label="Trạng thái Xuất MP3",
                                        value="Sẵn sàng.",
                                        interactive=False,
                                        lines=2,
                                    )
                                    vc_tts_out_audio = gr.Audio(
                                        label="Nghe trực tiếp File .MP3 vừa tạo",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    vc_tts_dl_mp3 = gr.File(
                                        label="Tải về File Audio (.mp3)",
                                    )

                # ---- Wire Tab 4 Events ----
                vc_clone_outputs = [vc_clone_table, vc_clone_status, vc_tts_clone_dd] + vc_slot_all_comps

                btn_vc_clone.click(
                    fn=vc_run_clone_voices,
                    inputs=[
                        vc_src_file,
                        vc_src_label_tb,
                        vc_src_num_spk_dd,
                        vc_src_thresh_sl,
                        vc_src_whisper_dd,
                        vc_src_pitch_dd,
                    ],
                    outputs=vc_clone_outputs,
                )

                btn_vc_refresh_bank.click(
                    fn=vc_refresh_bank,
                    outputs=vc_clone_outputs,
                )

                for _sid_comp, _sname_comp, _bren_comp, _bdel_comp in vc_slot_actions:
                    _bren_comp.click(
                        fn=vc_rename_clone_slot,
                        inputs=[_sid_comp, _sname_comp],
                        outputs=vc_clone_outputs,
                    )
                    _bdel_comp.click(
                        fn=vc_delete_clone_slot,
                        inputs=[_sid_comp],
                        outputs=vc_clone_outputs,
                    )

                btn_vc_analyze.click(
                    fn=vc_analyze_target_video,
                    inputs=[vc_tgt_video, vc_tgt_num_spk_dd, vc_tgt_whisper_dd],
                    outputs=[vc_target_speakers_state, vc_target_status] + vc_tgt_slot_all_comps,
                )

                btn_vc_suggest_f0.click(
                    fn=vc_suggest_mapping_f0,
                    inputs=[vc_target_speakers_state],
                    outputs=[vc_target_status] + vc_tgt_dd_list,
                )

                btn_vc_refresh_map.click(
                    fn=vc_refresh_target_dropdowns,
                    inputs=[vc_target_speakers_state] + vc_tgt_dd_list,
                    outputs=vc_tgt_dd_list,
                )

                vc_dub_run_event = btn_vc_dub_run.click(
                    fn=vc_run_dub_with_mapping,
                    inputs=[
                        vc_tgt_video,
                        vc_target_speakers_state,
                        vc_tgt_whisper_dd,
                        vc_tgt_num_spk_dd,
                        vc_tgt_gemini_dd,
                        vc_tgt_context_tb,
                        vc_tgt_api_key_tb,
                        vc_tgt_pitch_dd,
                        vc_tgt_pitch_shift_sl,
                    ] + vc_tgt_dd_list,
                    outputs=[vc_tgt_out_video, vc_tgt_dl_video, vc_tgt_dl_srt, vc_tgt_dub_status],
                )

                btn_vc_dub_cancel.click(
                    fn=vc_cancel_dub,
                    outputs=[vc_tgt_dub_status],
                    cancels=[vc_dub_run_event],
                )

                vc_tts_clone_dd.change(
                    fn=vc_preview_clone_for_tts,
                    inputs=[vc_tts_clone_dd],
                    outputs=[vc_tts_ref_preview, vc_tts_clone_info],
                )

                btn_vc_tts_run.click(
                    fn=vc_run_tts_export_mp3,
                    inputs=[
                        vc_tts_clone_dd,
                        vc_tts_text_tb,
                        vc_tts_lang_dd,
                        vc_tts_speed_sl,
                        vc_tts_pitch_dd,
                        vc_tts_pitch_shift_sl,
                    ],
                    outputs=[vc_tts_out_audio, vc_tts_dl_mp3, vc_tts_status],
                )

                demo.load(
                    fn=vc_refresh_bank,
                    outputs=vc_clone_outputs,
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
