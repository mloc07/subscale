"""Command-line interface for Eco Video Upscaler."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import Optional


import os


def _ensure_path_refreshed() -> None:
    """On Windows, re-read user/machine PATH from registry in case packages were just installed."""
    if sys.platform == "win32":
        try:
            import winreg
            paths = []
            for hkey, subkey in [
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, r"Environment"),
            ]:
                with winreg.OpenKey(hkey, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, "Path")
                    paths.append(val)
            combined = ";".join(paths)
            current = os.environ.get("PATH", "")
            for p in combined.split(";"):
                if p and p not in current:
                    current = f"{current};{p}"
            os.environ["PATH"] = current
        except Exception:
            pass


def check_ffmpeg() -> dict[str, any]:
    """Check FFmpeg and NVENC hardware encoder availability."""
    _ensure_path_refreshed()
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")

    info = {
        "ffmpeg": ffmpeg_bin is not None,
        "ffmpeg_path": ffmpeg_bin,
        "ffprobe": ffprobe_bin is not None,
        "ffprobe_path": ffprobe_bin,
        "version": None,
        "h264_nvenc": False,
        "hevc_nvenc": False,
    }

    if ffmpeg_bin:
        try:
            res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, check=False)
            if res.stdout:
                first_line = res.stdout.splitlines()[0] if res.stdout.splitlines() else ""
                info["version"] = first_line
        except Exception:
            pass

        try:
            res = subprocess.run([ffmpeg_bin, "-encoders"], capture_output=True, text=True, check=False)
            if res.stdout:
                info["h264_nvenc"] = "h264_nvenc" in res.stdout
                info["hevc_nvenc"] = "hevc_nvenc" in res.stdout
        except Exception:
            pass

    return info


def check_gpu() -> dict[str, any]:
    """Check NVIDIA GPU status using pynvml or nvidia-smi."""
    info = {
        "available": False,
        "count": 0,
        "name": None,
        "driver_version": None,
        "vram_total_mb": None,
        "temp_c": None,
        "power_w": None,
        "method": None,
    }

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except Exception:
                pwr = None
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8")

            info.update({
                "available": True,
                "count": count,
                "name": name,
                "driver_version": driver,
                "vram_total_mb": round(mem.total / (1024 * 1024)),
                "temp_c": temp,
                "power_w": round(pwr, 1) if pwr is not None else None,
                "method": "pynvml",
            })
            pynvml.nvmlShutdown()
            return info
    except Exception:
        pass

    # Fallback to nvidia-smi command line check
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    info.update({
                        "available": True,
                        "count": 1,
                        "name": parts[0],
                        "driver_version": parts[1],
                        "vram_total_mb": int(float(parts[2])),
                        "temp_c": int(float(parts[3])),
                        "power_w": float(parts[4]) if len(parts) > 4 and parts[4] != "[Not Supported]" else None,
                        "method": "nvidia-smi",
                    })
        except Exception:
            pass

    return info


def check_onnx() -> dict[str, any]:
    """Check ONNX Runtime and installed execution providers."""
    info = {
        "installed": False,
        "version": None,
        "providers": [],
        "dml_available": False,
        "cuda_available": False,
        "tensorrt_available": False,
    }

    try:
        import onnxruntime as ort

        info["installed"] = True
        info["version"] = ort.__version__
        providers = ort.get_available_providers()
        info["providers"] = providers
        info["dml_available"] = "DmlExecutionProvider" in providers
        info["cuda_available"] = "CUDAExecutionProvider" in providers
        info["tensorrt_available"] = "TensorrtExecutionProvider" in providers
    except ImportError:
        pass
    except Exception as e:
        info["error"] = str(e)

    return info


def run_doctor() -> int:
    """Print system diagnostics report."""
    print("==================================================")
    print(" Eco Video Upscaler (ecoup) - Environment Doctor ")
    print("==================================================")

    # 1. Platform
    print(f"\n[1] Python & OS:")
    print(f"  - Python Version : {sys.version.split()[0]}")
    print(f"  - Executable     : {sys.executable}")
    print(f"  - Platform       : {sys.platform}")

    # 2. FFmpeg
    ff = check_ffmpeg()
    print(f"\n[2] FFmpeg & Video Tools:")
    if ff["ffmpeg"]:
        print(f"  - FFmpeg         : [OK] Found at {ff['ffmpeg_path']}")
        if ff["version"]:
            print(f"  - Version        : {ff['version']}")
        print(f"  - FFprobe        : [{'OK' if ff['ffprobe'] else 'MISSING'}] {ff['ffprobe_path'] or ''}")
        print(f"  - NVENC H.264    : [{'OK' if ff['h264_nvenc'] else 'NOT SUPPORTED'}]")
        print(f"  - NVENC HEVC     : [{'OK' if ff['hevc_nvenc'] else 'NOT SUPPORTED'}]")
    else:
        print("  - FFmpeg         : [WARNING] ffmpeg not found in PATH!")
        print("                     Please install FFmpeg (e.g. winget install Gyan.FFmpeg)")

    # 3. GPU
    gpu = check_gpu()
    print(f"\n[3] NVIDIA GPU & Thermal Access:")
    if gpu["available"]:
        print(f"  - GPU Name       : [OK] {gpu['name']} (Count: {gpu['count']})")
        print(f"  - Driver Version : {gpu['driver_version']}")
        print(f"  - VRAM Total     : {gpu['vram_total_mb']} MB")
        print(f"  - Current Temp   : {gpu['temp_c']} °C")
        if gpu["power_w"] is not None:
            print(f"  - Current Power  : {gpu['power_w']} W")
        print(f"  - Monitor Method : {gpu['method']}")
    else:
        print("  - GPU            : [INFO] No NVIDIA GPU or telemetry accessible.")
        print("                     Software fallback will use CPU.")

    # 4. ONNX Runtime
    onnx = check_onnx()
    print(f"\n[4] AI Inference (ONNX Runtime):")
    if onnx["installed"]:
        print(f"  - ONNX Runtime   : [OK] Version {onnx['version']}")
        print(f"  - Providers      : {', '.join(onnx['providers'])}")
        print(f"  - DirectML EP    : [{'OK (DirectX 12 GPU)' if onnx['dml_available'] else 'NOT AVAILABLE'}]")
        print(f"  - CUDA EP        : [{'OK' if onnx['cuda_available'] else 'NOT AVAILABLE'}]")
        print(f"  - TensorRT EP    : [{'OK' if onnx['tensorrt_available'] else 'NOT AVAILABLE'}]")
    else:
        print("  - ONNX Runtime   : [WARNING] onnxruntime not installed in this environment.")
        print("                     Run: pip install onnxruntime-directml")

    print("\n==================================================")
    all_good = ff["ffmpeg"] and (gpu["available"] or True)
    if all_good:
        print(" Diagnostic status: READY FOR PIPELINE")
    else:
        print(" Diagnostic status: SOME COMPONENTS NEED SETUP")
    print("==================================================\n")
    return 0


def main(args: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Eco Video Upscaler CLI")
    subparsers = parser.add_subparsers(dest="command")

    # doctor command
    subparsers.add_parser("doctor", help="Inspect environment and hardware capability")

    # placeholder run command
    run_parser = subparsers.add_parser("run", help="Upscale a video file")
    run_parser.add_argument("-i", "--input", help="Path to input video file")
    run_parser.add_argument("-o", "--output", help="Path to output video file")
    run_parser.add_argument("-c", "--config", default="config.yaml", help="Path to config.yaml")
    run_parser.add_argument("--scale", type=int, help="Output target scale factor")
    run_parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    # subtitle command
    sub_parser = subparsers.add_parser("subtitle", help="Auto subtitles: Whisper -> Gemini translation -> burn-in")
    sub_parser.add_argument("-i", "--input", required=True, help="Path to input video")
    sub_parser.add_argument("-o", "--output", help="Output video path (default: <input>_sub.mp4)")
    sub_parser.add_argument("--out-dir", help="Directory for .srt/.ass/report files (default: next to output)")
    sub_parser.add_argument("--lang", default="auto", help="Source language code or 'auto'")
    sub_parser.add_argument("--target", default="vi", help="Target language code (default: vi)")
    sub_parser.add_argument("--whisper-model", default="large-v3")
    sub_parser.add_argument("--translate-model", default=None, help="Gemini model id (default from config)")
    sub_parser.add_argument("--context-note", default="", help="Free-form context for the translator")
    sub_parser.add_argument("--glossary", default=None, help="JSON file {\"term\": \"translation\"}")
    sub_parser.add_argument("--no-burn", action="store_true", help="Only write .srt files, do not burn into video")
    sub_parser.add_argument("--show-original", action="store_true", help="Show source text above the translation")
    sub_parser.add_argument("--font", default=None, help="Font family for burn-in (must support Vietnamese)")
    sub_parser.add_argument("--cache-dir", default=None, help="Translation cache directory (enables resume)")

    # dubbing command
    dub_parser = subparsers.add_parser("dub", help="AI Multi-speaker Dubbing into Vietnamese")
    dub_parser.add_argument("-i", "--input", required=True, help="Path to input video")
    dub_parser.add_argument("-o", "--project-dir", help="Project directory (default: <input_dir>/dub_project_<stem>)")
    dub_parser.add_argument("--backend", default="openvoice", choices=["openvoice", "clone", "f5", "gtts", "mock"], help="TTS Voice backend")
    dub_parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    dub_parser.add_argument("--translate-model", default="gemini-2.5-flash", help="Gemini translation model")
    dub_parser.add_argument("--num-speakers", type=int, default=None, help="Number of speakers (optional)")
    dub_parser.add_argument("--context", default="", help="Context notes for translator/dubbing style")
    dub_parser.add_argument("--voice-map", default=None, help='Speaker mapping JSON/file or k=v list (e.g. "spk_0=clone1,spk_1=clone2")')
    dub_parser.add_argument("--voice-bank", default=None, help="Directory of cloned Voice Bank")
    dub_parser.add_argument("--min-speed", type=float, default=0.85, help="Minimum alignment speed factor (0.5..1.0)")
    dub_parser.add_argument("--pitch-mode", default="auto", choices=["auto", "none", "manual"], help="Pitch alignment mode")
    dub_parser.add_argument("--pitch-shift", type=float, default=0.0, help="Manual pitch shift semitones")
    dub_parser.add_argument("--mock", action="store_true", help="Run in mock mode for offline/unit tests")
    dub_parser.add_argument("--step", default=None, choices=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"], help="Run up to specific step only")

    # clone-voices command
    clone_parser = subparsers.add_parser("clone-voices", help="Extract & clone all speaker voices from video into Voice Bank")
    clone_parser.add_argument("-i", "--input", required=True, help="Path to source video/audio")
    clone_parser.add_argument("--bank-dir", default=None, help="Directory of Voice Bank")
    clone_parser.add_argument("--label", default="", help="Display label prefix for cloned voices")
    clone_parser.add_argument("--num-speakers", type=int, default=None, help="Expected number of speakers")
    clone_parser.add_argument("--whisper-model", default="medium", help="Whisper model size")
    clone_parser.add_argument("--distance-threshold", type=float, default=0.72, help="Diarization distance threshold")
    clone_parser.add_argument("--pitch-mode", default="auto", choices=["auto", "none", "manual"], help="Pitch alignment mode")
    clone_parser.add_argument("--mock", action="store_true", help="Run in mock mode")

    # tts-clone command
    tts_clone_parser = subparsers.add_parser("tts-clone", help="Read text using a cloned voice and export .mp3")
    tts_clone_parser.add_argument("--clone", required=True, help="Clone ID in Voice Bank or path to reference .wav")
    tts_clone_parser.add_argument("--text", default="", help="Text to synthesize")
    tts_clone_parser.add_argument("--text-file", default=None, help="UTF-8 text file to synthesize")
    tts_clone_parser.add_argument("-o", "--output", required=True, help="Output .mp3 path")
    tts_clone_parser.add_argument("--bank-dir", default=None, help="Directory of Voice Bank")
    tts_clone_parser.add_argument("--lang", default="vi", help="TTS language (default: vi)")
    tts_clone_parser.add_argument("--speed", type=float, default=1.0, help="Reading speed (0.6..1.8)")
    tts_clone_parser.add_argument("--pitch-mode", default="auto", choices=["auto", "none", "manual"], help="Pitch alignment mode")
    tts_clone_parser.add_argument("--pitch-shift", type=float, default=0.0, help="Manual pitch shift semitones")
    tts_clone_parser.add_argument("--mock", action="store_true", help="Run in mock mode")

    parsed = parser.parse_args(args)

    if parsed.command == "doctor":
        return run_doctor()
    elif parsed.command == "subtitle":
        import json
        from pathlib import Path
        from ecoup.subtitle.config import SubtitleConfig
        from ecoup.subtitle.pipeline import SubtitlePipeline

        cfg = SubtitleConfig()
        cfg.asr.model = parsed.whisper_model
        cfg.asr.language = None if parsed.lang == "auto" else parsed.lang
        cfg.translate.target_language = parsed.target
        if parsed.translate_model:
            cfg.translate.model = parsed.translate_model
        cfg.translate.context_note = parsed.context_note
        cfg.translate.cache_dir = parsed.cache_dir
        if parsed.glossary:
            cfg.translate.glossary = json.loads(Path(parsed.glossary).read_text(encoding="utf-8"))
        cfg.burn_in = not parsed.no_burn
        cfg.burn.show_original = parsed.show_original
        if parsed.font:
            cfg.burn.font_name = parsed.font

        def _on_progress(step: str, frac: float) -> None:
            print(f"[subtitle] {step}: {int(frac * 100)}%", flush=True)

        pipe = SubtitlePipeline(cfg, on_progress=_on_progress)
        try:
            rep = pipe.run(parsed.input, output=parsed.output, out_dir=parsed.out_dir)
        except KeyboardInterrupt:
            pipe.cancel()
            print("[subtitle] cancelled")
            return 130
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
        return 0
    elif parsed.command == "dub":
        import json
        from pathlib import Path
        from ecoup.dubbing.config import DubbingConfig
        from ecoup.dubbing.pipeline import DubbingPipeline

        cfg = DubbingConfig()
        if parsed.project_dir:
            cfg.project_dir = parsed.project_dir
        cfg.tts.backend = parsed.backend
        cfg.asr.model_size = parsed.whisper_model
        cfg.translation.model = parsed.translate_model
        cfg.tts.min_speed = parsed.min_speed
        cfg.tts.pitch_match_mode = parsed.pitch_mode
        cfg.tts.pitch_shift_semitones = parsed.pitch_shift
        if parsed.voice_bank:
            cfg.tts.voice_bank_dir = parsed.voice_bank
        if parsed.mock or parsed.backend == "mock":
            cfg.demucs.mock = True
            cfg.asr.mock = True
            cfg.diarization.mock = True
            cfg.translation.mock = True
            cfg.tts.mock = True
        if parsed.num_speakers:
            cfg.diarization.num_speakers = parsed.num_speakers
        if parsed.voice_map:
            raw_map = parsed.voice_map.strip()
            map_path = Path(raw_map)
            if map_path.is_file():
                cfg.tts.speaker_voice_map = json.loads(map_path.read_text(encoding="utf-8"))
            elif raw_map.startswith("{"):
                cfg.tts.speaker_voice_map = json.loads(raw_map)
            else:
                parsed_map = {}
                for pair in raw_map.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        parsed_map[k.strip()] = v.strip()
                cfg.tts.speaker_voice_map = parsed_map

        def _on_dub_progress(step: str, frac: float) -> None:
            print(f"[dubbing] {step}: {int(frac * 100)}%", flush=True)

        pipeline = DubbingPipeline(parsed.input, config=cfg, on_progress=_on_dub_progress)

        if parsed.step == "s0":
            res = pipeline.run_s0_intake()
            print(f"[dubbing] S0 complete: {res}")
        elif parsed.step == "s1":
            d, b = pipeline.run_s1_separation()
            print(f"[dubbing] S1 complete: dialogue={d}, bgm={b}")
        elif parsed.step == "s2":
            segs = pipeline.run_s2_asr()
            print(f"[dubbing] S2 complete: {len(segs)} segments transcribed.")
        elif parsed.step == "s3":
            segs = pipeline.run_s3_diarization()
            print(f"[dubbing] S3 complete: {len(segs)} segments diarized.")
        elif parsed.step == "s4":
            profs = pipeline.run_s4_voice_profiles()
            print(f"[dubbing] S4 complete: {len(profs)} voice profiles.")
        elif parsed.step == "s5":
            trans = pipeline.run_s5_translation(context_notes=parsed.context)
            print(f"[dubbing] S5 complete: {len(trans)} segments translated.")
        else:
            res = pipeline.run_full_pipeline(context_notes=parsed.context)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    elif parsed.command == "clone-voices":
        import json
        from dataclasses import asdict
        from ecoup.dubbing.voice_bank import clone_voices_from_video

        def _on_clone_progress(step: str, frac: float) -> None:
            print(f"[clone-voices] {step}: {int(frac * 100)}%", flush=True)

        entries = clone_voices_from_video(
            video_path=parsed.input,
            bank_dir=parsed.bank_dir,
            label_prefix=parsed.label,
            num_speakers=parsed.num_speakers,
            whisper_model=parsed.whisper_model,
            distance_threshold=parsed.distance_threshold,
            pitch_match_mode=parsed.pitch_mode,
            mock=parsed.mock,
            on_progress=_on_clone_progress,
        )
        print(json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2))
        return 0
    elif parsed.command == "tts-clone":
        import json
        from pathlib import Path
        from ecoup.dubbing.tts import MockVoiceBackend, OpenVoiceBackend, synthesize_text_to_mp3
        from ecoup.dubbing.voice_bank import VoiceBankManager

        text_val = (parsed.text or "").strip()
        if not text_val and parsed.text_file:
            text_val = Path(parsed.text_file).read_text(encoding="utf-8").strip()
        if not text_val:
            print("[ERROR] Either --text or --text-file is required.")
            return 1

        bank = VoiceBankManager(parsed.bank_dir)
        ref_path = bank.resolve_voice(parsed.clone)
        if ref_path is None or not ref_path.exists():
            print(f"[ERROR] Cloned voice not found: {parsed.clone}")
            return 1

        backend = (
            MockVoiceBackend()
            if parsed.mock
            else OpenVoiceBackend(
                base_lang=parsed.lang,
                pitch_match_mode=parsed.pitch_mode,
                pitch_shift_semitones=parsed.pitch_shift,
            )
        )
        rep = synthesize_text_to_mp3(
            text=text_val,
            output_mp3=parsed.output,
            backend=backend,
            ref_audio_path=ref_path,
            speed=parsed.speed,
            speaker_id=parsed.clone,
        )
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0
    elif parsed.command == "run":
        from pathlib import Path
        from ecoup.config import load_config
        from ecoup.engine.onnx_engine import OnnxUpscaleEngine
        from ecoup.pipeline import UpscalePipeline
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

        cfg = load_config(parsed.config)
        input_file = parsed.input or cfg.input
        output_file = parsed.output or cfg.output

        if not input_file:
            print("[ERROR] Input video path is required. Use -i <path> or set 'input' in config.yaml.")
            return 1
        if not output_file:
            # Default output: input_upscaled.mp4
            p = Path(input_file)
            output_file = str(p.parent / f"{p.stem}_upscaled{p.suffix}")

        if parsed.scale:
            cfg.model.target_scale = parsed.scale

        # Initialize ONNX engine if model exists
        engine = None
        model_path = Path(cfg.model.path)
        if model_path.exists():
            engine = OnnxUpscaleEngine()
            engine.load(model_path, precision=cfg.model.precision, provider=cfg.inference.provider)
        else:
            print(f"[ERROR] ONNX model not found at '{model_path}'.")
            print("Please place the model file in 'models/' or update config.yaml.")
            return 1

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[fps]:.1f} fps[/cyan]"),
            TextColumn("•"),
            TextColumn("[yellow]{task.fields[temp]}°C[/yellow]"),
            TextColumn("[blue]{task.fields[power]}W[/blue]"),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Upscaling", total=100, fps=0.0, temp="--", power="--")

            def on_progress(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate):
                progress.update(
                    task_id,
                    completed=frame_idx,
                    total=total_frames,
                    fps=fps,
                    temp=str(metrics.temp_c) if metrics else "--",
                    power=str(round(metrics.power_w, 1)) if metrics and metrics.power_w else "--",
                )

            pipeline = UpscalePipeline(config=cfg, engine=engine, on_progress=on_progress)
            try:
                pipeline.run(input_path=input_file, output_path=output_file, resume=parsed.resume)
            finally:
                if engine:
                    engine.close()

        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
