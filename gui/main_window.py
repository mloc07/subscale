"""PyQt6 GUI Main Window for Eco Video Upscaler & Subtitle Studio."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
    QAbstractItemView,
)

from ecoup.cli import check_gpu, _ensure_path_refreshed
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
from ecoup.video_io import get_video_metadata, VideoMetadata


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #18181b;
    color: #e4e4e7;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2e2e33;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    background-color: #202023;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 5px 8px;
    color: #f4f4f5;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #3b82f6;
}
QPushButton {
    background-color: #2e2e33;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    color: #f4f4f5;
}
QPushButton:hover {
    background-color: #3f3f46;
}
QPushButton#primaryBtn {
    background-color: #2563eb;
    border: 1px solid #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #1d4ed8;
}
QPushButton#dangerBtn {
    color: #ef4444;
}
QPushButton#dangerBtn:hover {
    background-color: #451a1a;
    border-color: #ef4444;
}
QProgressBar {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    height: 12px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 3px;
}
QTextEdit {
    background-color: #121214;
    border: 1px solid #27272a;
    border-radius: 4px;
    color: #a1a1aa;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}
QTabWidget::pane {
    border: 1px solid #2e2e33;
    border-radius: 6px;
    background-color: #18181b;
}
QTabBar::tab {
    background-color: #202023;
    border: 1px solid #2e2e33;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 16px;
    margin-right: 3px;
    color: #a1a1aa;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #27272a;
    border-color: #3b82f6;
    color: #3b82f6;
    font-weight: 600;
}
QTabBar::tab:hover {
    color: #f4f4f5;
}
"""


class PipelineWorker(QThread):
    """Background worker thread for running the upscale pipeline."""
    progress_sig = pyqtSignal(int, int, float, float, object, float)
    log_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, str)

    def __init__(self, config: AppConfig, input_path: str, output_path: str, resume: bool = False):
        super().__init__()
        self.config = config
        self.input_path = input_path
        self.output_path = output_path
        self.resume = resume
        self.pipeline: Optional[UpscalePipeline] = None
        self.engine: Optional[OnnxUpscaleEngine] = None

    def run(self):
        try:
            model_path = Path(self.config.model.path)
            custom_fn = None
            if model_path.exists():
                self.log_sig.emit(f"Đang nạp mô hình AI: {model_path.name}...")
                self.engine = OnnxUpscaleEngine()
                self.engine.load(model_path, precision=self.config.model.precision, provider=self.config.inference.provider)
                if hasattr(self.engine, "scale") and self.engine.scale > 0:
                    self.config.model.scale = self.engine.scale
                    self.log_sig.emit(f"Mô hình đã sẵn sàng (Native Scale: {self.engine.scale}x).")
            else:
                self.log_sig.emit(f"[Thông báo] Không tìm thấy file model AI '{model_path.name}'. Đang chạy chế độ nội suy Lanczos4 chất lượng cao...")
                import cv2
                scale = self.config.model.scale
                def lanczos_fallback(tile):
                    th, tw = tile.shape[:2]
                    return cv2.resize(tile, (tw * scale, th * scale), interpolation=cv2.INTER_LANCZOS4)
                custom_fn = lanczos_fallback

            def on_progress(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate):
                self.progress_sig.emit(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate)

            self.pipeline = UpscalePipeline(
                config=self.config,
                engine=self.engine,
                custom_upscale_fn=custom_fn,
                on_progress=on_progress,
            )

            self.log_sig.emit(f"Bắt đầu xử lý: {Path(self.input_path).name} -> {Path(self.output_path).name}")
            self.pipeline.run(self.input_path, self.output_path, resume=self.resume)
            self.finished_sig.emit(True, "Hoàn tất upscale video thành công!")
        except Exception as e:
            self.finished_sig.emit(False, str(e))
        finally:
            if self.engine:
                self.engine.close()

    def cancel(self):
        if self.pipeline:
            self.pipeline.cancel()


class SubtitleWorker(QThread):
    """Background worker thread for running the subtitle pipeline."""
    progress_sig = pyqtSignal(str, float)
    log_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, object, str)

    def __init__(self, config, input_path: str, output_path: str, out_dir: str):
        super().__init__()
        self.config = config
        self.input_path = input_path
        self.output_path = output_path
        self.out_dir = out_dir
        self.pipeline = None

    def run(self):
        try:
            from ecoup.subtitle.pipeline import SubtitlePipeline

            step_names = {
                "audio": "Tách âm thanh",
                "asr": "Nhận dạng giọng nói (Whisper)",
                "normalize": "Chuẩn hóa câu thoại",
                "translate": "Dịch tiếng Việt (Gemini)",
                "srt": "Xuất phụ đề SRT",
                "burn": "Ghép cứng video (FFmpeg libass)",
            }

            def on_progress(step: str, frac: float):
                name = step_names.get(step, step)
                self.progress_sig.emit(step, frac)
                self.log_sig.emit(f"[Phụ đề - {name}] Tiến độ: {int(frac * 100)}%")

            self.pipeline = SubtitlePipeline(config=self.config, on_progress=on_progress)
            self.log_sig.emit(f"Bắt đầu quy trình phụ đề cho: {Path(self.input_path).name}")
            rep = self.pipeline.run(self.input_path, output=self.output_path, out_dir=self.out_dir)

            if rep.cancelled:
                self.finished_sig.emit(False, rep, "Tiến trình tạo phụ đề đã bị hủy.")
            else:
                self.finished_sig.emit(True, rep, f"Tạo phụ đề thành công! ({rep.cues} câu thoại)")
        except Exception as e:
            self.finished_sig.emit(False, None, str(e))

    def cancel(self):
        if self.pipeline:
            self.pipeline.cancel()


class DubbingWorker(QThread):
    """Background worker thread for running the multi-speaker dubbing pipeline."""
    progress_sig = pyqtSignal(str, float)
    log_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, dict, str)

    def __init__(self, config, input_path: str, context_notes: str = "", target_step: Optional[str] = None):
        super().__init__()
        self.config = config
        self.input_path = input_path
        self.context_notes = context_notes
        self.target_step = target_step
        self.pipeline = None

    def run(self):
        try:
            from ecoup.dubbing.pipeline import DubbingPipeline

            step_names = {
                "S0": "Kiểm tra âm thanh",
                "S1": "Tách nhạc nền & thoại (Demucs)",
                "S2": "Nhận dạng giọng nói (Whisper)",
                "S3": "Phân vai nhân vật (SpeechBrain)",
                "S4": "Trích xuất mẫu giọng chuẩn",
                "S5": "Dịch thuật khớp thời lượng (Gemini)",
                "S6": "Tổng hợp tiếng Việt (TTS)",
                "S7": "Khớp thời lượng (Snapping)",
                "S8": "Trộn âm & Ducking nhạc nền",
                "S9": "Ghép video đầu ra",
            }

            def on_progress(step: str, frac: float):
                name = step_names.get(step, step)
                self.progress_sig.emit(step, frac)
                self.log_sig.emit(f"[Lồng tiếng - {name}] Bước {step} ({int(frac * 100)}%)")

            self.pipeline = DubbingPipeline(self.input_path, config=self.config, on_progress=on_progress)
            self.log_sig.emit(f"Bắt đầu quy trình lồng tiếng cho: {Path(self.input_path).name}")

            if self.target_step:
                res = self.pipeline.run_step(self.target_step, context_notes=self.context_notes)
            else:
                res = self.pipeline.run_full_pipeline(context_notes=self.context_notes)

            if getattr(self.pipeline, "_cancelled", False):
                self.finished_sig.emit(False, res, "Tiến trình lồng tiếng đã bị hủy.")
            else:
                self.finished_sig.emit(True, res, "Hoàn tất lồng tiếng video thành công!")
        except Exception as e:
            self.finished_sig.emit(False, {}, str(e))

    def cancel(self):
        if self.pipeline:
            self.pipeline.cancel()


class CloneVoicesWorker(QThread):
    """Background worker to clone all speaker voices from a source video/audio."""
    progress_sig = pyqtSignal(str, float)
    log_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, list, str)

    def __init__(
        self,
        source_path: str,
        label_prefix: str = "",
        num_speakers: Optional[int] = None,
        distance_threshold: float = 0.72,
        whisper_model: str = "medium",
        pitch_match_mode: str = "auto",
    ):
        super().__init__()
        self.source_path = source_path
        self.label_prefix = label_prefix
        self.num_speakers = num_speakers
        self.distance_threshold = distance_threshold
        self.whisper_model = whisper_model
        self.pitch_match_mode = pitch_match_mode

    def run(self):
        try:
            from ecoup.dubbing.voice_bank import clone_voices_from_video

            def _on_prog(step: str, frac: float):
                self.progress_sig.emit(step, frac)
                self.log_sig.emit(f"[Clone Giọng - {step}] {int(frac * 100)}%")

            entries = clone_voices_from_video(
                video_path=self.source_path,
                label_prefix=self.label_prefix,
                num_speakers=self.num_speakers,
                distance_threshold=self.distance_threshold,
                whisper_model=self.whisper_model,
                pitch_match_mode=self.pitch_match_mode,
                on_progress=_on_prog,
            )
            self.finished_sig.emit(True, entries, f"Đã clone thành công {len(entries)} giọng nói.")
        except Exception as e:
            self.finished_sig.emit(False, [], str(e))


class AnalyzeTargetSpeakersWorker(QThread):
    """Background worker to detect characters/speakers in a target video."""
    progress_sig = pyqtSignal(str, float)
    log_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, list, str)

    def __init__(self, target_video: str, num_speakers: Optional[int], whisper_model: str):
        super().__init__()
        self.target_video = target_video
        self.num_speakers = num_speakers
        self.whisper_model = whisper_model

    def run(self):
        try:
            from ecoup.dubbing.config import DubbingConfig
            from ecoup.dubbing.pipeline import DubbingPipeline

            p = Path(self.target_video)
            cfg = DubbingConfig(project_dir=str(p.parent / f"dub_project_{p.stem}"))
            cfg.asr.model_size = self.whisper_model
            cfg.diarization.num_speakers = self.num_speakers

            def _on_prog(step: str, frac: float):
                self.progress_sig.emit(step, frac)
                self.log_sig.emit(f"[Phân tích Video đích - {step}] {int(frac * 100)}%")

            pipe = DubbingPipeline(self.target_video, config=cfg, on_progress=_on_prog)
            speakers = pipe.analyze_target_speakers()
            self.finished_sig.emit(True, speakers, f"Phát hiện {len(speakers)} nhân vật trong video đích.")
        except Exception as e:
            self.finished_sig.emit(False, [], str(e))


class TtsCloneMp3Worker(QThread):
    """Background worker to synthesize text using a cloned voice and export .mp3."""
    progress_sig = pyqtSignal(float)
    log_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, dict, str)

    def __init__(
        self,
        clone_spec: str,
        text: str,
        output_mp3: str,
        base_lang: str = "vi",
        speed: float = 1.0,
        pitch_match_mode: str = "auto",
        pitch_shift_semitones: float = 0.0,
    ):
        super().__init__()
        self.clone_spec = clone_spec
        self.text = text
        self.output_mp3 = output_mp3
        self.base_lang = base_lang
        self.speed = speed
        self.pitch_match_mode = pitch_match_mode
        self.pitch_shift_semitones = pitch_shift_semitones

    def run(self):
        backend = None
        try:
            from ecoup.dubbing.tts import OpenVoiceBackend, synthesize_text_to_mp3
            from ecoup.dubbing.voice_bank import VoiceBankManager

            bank = VoiceBankManager()
            ref_path = bank.resolve_voice(self.clone_spec)
            if ref_path is None or not ref_path.exists():
                self.finished_sig.emit(False, {}, f"Không tìm thấy mẫu giọng '{self.clone_spec}'.")
                return

            clone_id = VoiceBankManager._normalize_choice_id(self.clone_spec)
            backend = OpenVoiceBackend(
                device="auto",
                base_lang=self.base_lang,
                pitch_match_mode=self.pitch_match_mode,
                pitch_shift_semitones=self.pitch_shift_semitones,
            )

            def _on_prog(frac: float):
                self.progress_sig.emit(frac)
                self.log_sig.emit(f"[TTS Xuất MP3] {int(frac * 100)}%")

            rep = synthesize_text_to_mp3(
                text=self.text,
                output_mp3=self.output_mp3,
                backend=backend,
                ref_audio_path=ref_path,
                speed=self.speed,
                speaker_id=clone_id,
                bitrate="192k",
                on_progress=_on_prog,
            )
            self.finished_sig.emit(True, rep, "Đã tạo file .mp3 thành công!")
        except Exception as e:
            self.finished_sig.emit(False, {}, str(e))
        finally:
            if backend is not None and getattr(backend, "_converter", None) is not None:
                try:
                    backend._converter = None
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass


class MainWindow(QMainWindow):
    """Main window for Eco Video Upscaler & Subtitle Studio."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eco Video Upscaler & Subtitle Studio")
        self.setMinimumSize(820, 720)
        self.setStyleSheet(DARK_STYLE)

        self.worker: Optional[PipelineWorker] = None
        self.sub_worker: Optional[SubtitleWorker] = None
        self.dub_worker: Optional[DubbingWorker] = None
        self.vc_clone_worker: Optional[CloneVoicesWorker] = None
        self.vc_analyze_worker: Optional[AnalyzeTargetSpeakersWorker] = None
        self.vc_dub_worker: Optional[DubbingWorker] = None
        self.vc_tts_worker: Optional[TtsCloneMp3Worker] = None
        self.vc_target_speakers: list[dict] = []
        self.vc_target_combos: dict[str, QComboBox] = {}
        self.is_paused = False
        self.meta: Optional[VideoMetadata] = None
        self.sub_meta: Optional[VideoMetadata] = None
        self.dub_meta: Optional[VideoMetadata] = None

        self._build_ui()
        self._init_hardware_info()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(16, 14, 16, 14)

        # 1. Header Row
        header_layout = QHBoxLayout()
        title_label = QLabel("Eco Video Upscaler & Subtitle Studio")
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.status_badge = QLabel("Sẵn sàng")
        self.status_badge.setStyleSheet("color: #10b981; font-weight: 500;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)
        root_layout.addLayout(header_layout)

        # 2. Main Tabs
        self.tabs = QTabWidget()
        upscale_tab = self._build_upscale_tab()
        subtitle_tab = self._build_subtitle_tab()
        dubbing_tab = self._build_dubbing_tab()
        voice_clone_tab = self._build_voice_clone_tab()
        self.tabs.addTab(upscale_tab, "⚡ Nâng cấp Video (Upscale)")
        self.tabs.addTab(subtitle_tab, "📝 Phụ đề tự động (Subtitle)")
        self.tabs.addTab(dubbing_tab, "🎙️ Lồng tiếng AI (Dubbing)")
        self.tabs.addTab(voice_clone_tab, "🧬 Voice Cloning Studio")
        root_layout.addWidget(self.tabs)

        # 3. Real-time Console Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(100)
        self.log_box.append("[Hệ thống] Giao diện Eco Video Upscaler & Subtitle Studio đã sẵn sàng.")
        root_layout.addWidget(self.log_box)

    def _build_upscale_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 10, 8, 8)

        # File Selection Panel
        file_box = QWidget()
        file_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        file_layout = QGridLayout(file_box)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(8)

        file_layout.addWidget(QLabel("Video vào:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Chọn video nguồn (.mp4, .mkv, .mov)...")
        file_layout.addWidget(self.input_edit, 0, 1)
        browse_in_btn = QPushButton("Chọn...")
        browse_in_btn.clicked.connect(self._browse_input)
        file_layout.addWidget(browse_in_btn, 0, 2)

        file_layout.addWidget(QLabel("Video ra:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Đường dẫn lưu video sau khi upscale...")
        file_layout.addWidget(self.output_edit, 1, 1)
        browse_out_btn = QPushButton("Lưu tại...")
        browse_out_btn.clicked.connect(self._browse_output)
        file_layout.addWidget(browse_out_btn, 1, 2)

        self.meta_label = QLabel("Chưa chọn video")
        self.meta_label.setStyleSheet("color: #71717a; font-family: monospace; font-size: 11px;")
        file_layout.addWidget(self.meta_label, 2, 1, 1, 2)
        layout.addWidget(file_box)

        # Settings Panel
        settings_box = QWidget()
        settings_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        set_layout = QVBoxLayout(settings_box)
        set_layout.setContentsMargins(10, 10, 10, 10)
        set_layout.setSpacing(8)

        dropdown_row = QHBoxLayout()
        dropdown_row.setSpacing(10)

        m_col = QVBoxLayout()
        m_col.addWidget(QLabel("Mô hình AI:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "NomosUni 2x (Siêu nét, video đời thực - Nhẹ 2.3MB)",
            "SPAN 2x (Siêu nhanh, mát GPU - Nhẹ 1.6MB)",
            "realesr-general-x4v3 (4x, Khử nhiễu mịn da - 4.9MB)",
            "realesr-animevideov3 (4x, Hoạt hình / 2D - 2.5MB)",
            "RealESRGAN x2plus (2x, Chi tiết cao cấp - Nặng 67MB)",
        ])
        m_col.addWidget(self.model_combo)
        dropdown_row.addLayout(m_col)

        s_col = QVBoxLayout()
        s_col.addWidget(QLabel("Độ phân giải đích:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "1080p (Scale 2x)",
            "1440p (Scale 2x)",
            "4K (Scale 4x)",
        ])
        s_col.addWidget(self.scale_combo)
        dropdown_row.addLayout(s_col)

        e_col = QVBoxLayout()
        e_col.addWidget(QLabel("Bộ mã hóa:"))
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems([
            "NVIDIA NVENC H.264 (Phần cứng)",
            "NVIDIA NVENC HEVC / H.265",
            "x264 (CPU)",
        ])
        e_col.addWidget(self.encoder_combo)
        dropdown_row.addLayout(e_col)

        t_col = QVBoxLayout()
        t_col.addWidget(QLabel("Chia Tile:"))
        self.tile_combo = QComboBox()
        self.tile_combo.addItems([
            "256x256 (Mát máy / Tiết kiệm VRAM)",
            "512x512 (Cân bằng)",
            "Toàn khung (Nhanh nhất)",
        ])
        t_col.addWidget(self.tile_combo)
        dropdown_row.addLayout(t_col)
        set_layout.addLayout(dropdown_row)

        chk_row = QHBoxLayout()
        self.thermal_chk = QCheckBox("Kiểm soát nhiệt độ:")
        self.thermal_chk.setChecked(True)
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(50, 90)
        self.temp_spin.setValue(70)
        self.temp_spin.setSuffix(" °C")

        self.dedup_chk = QCheckBox("Bỏ qua khung trùng")
        self.dedup_chk.setChecked(True)
        self.checkpoint_chk = QCheckBox("Lưu checkpoint định kỳ")
        self.checkpoint_chk.setChecked(True)

        chk_row.addWidget(self.thermal_chk)
        chk_row.addWidget(self.temp_spin)
        chk_row.addSpacing(14)
        chk_row.addWidget(self.dedup_chk)
        chk_row.addSpacing(14)
        chk_row.addWidget(self.checkpoint_chk)
        chk_row.addStretch()
        set_layout.addLayout(chk_row)

        realism_row = QHBoxLayout()
        realism_row.setSpacing(12)

        d_box = QHBoxLayout()
        d_box.addWidget(QLabel("Khử nhiễu:"))
        self.pre_denoise_spin = QSpinBox()
        self.pre_denoise_spin.setRange(0, 100)
        self.pre_denoise_spin.setValue(30)
        self.pre_denoise_spin.setSuffix(" %")
        d_box.addWidget(self.pre_denoise_spin)
        realism_row.addLayout(d_box)

        r_box = QHBoxLayout()
        r_box.addWidget(QLabel("Khôi phục:"))
        self.recover_spin = QSpinBox()
        self.recover_spin.setRange(0, 50)
        self.recover_spin.setValue(20)
        self.recover_spin.setSuffix(" %")
        r_box.addWidget(self.recover_spin)
        realism_row.addLayout(r_box)

        s_box = QHBoxLayout()
        s_box.addWidget(QLabel("Làm nét:"))
        self.sharpen_spin = QSpinBox()
        self.sharpen_spin.setRange(0, 100)
        self.sharpen_spin.setValue(20)
        self.sharpen_spin.setSuffix(" %")
        s_box.addWidget(self.sharpen_spin)
        realism_row.addLayout(s_box)

        self.grain_chk = QCheckBox("Bù hạt (Grain)")
        self.grain_chk.setChecked(True)
        realism_row.addWidget(self.grain_chk)
        realism_row.addStretch()
        set_layout.addLayout(realism_row)
        layout.addWidget(settings_box)

        # Hardware Monitor Box
        hw_box = QWidget()
        hw_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px;")
        hw_layout = QGridLayout(hw_box)
        hw_layout.setContentsMargins(10, 8, 10, 8)

        hw_layout.addWidget(QLabel("Nhiệt độ GPU"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        self.val_temp = QLabel("-- °C")
        self.val_temp.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.val_temp.setStyleSheet("color: #10b981;")
        hw_layout.addWidget(self.val_temp, 1, 0, Qt.AlignmentFlag.AlignCenter)

        hw_layout.addWidget(QLabel("Công suất"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        self.val_power = QLabel("-- W")
        self.val_power.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.val_power.setStyleSheet("color: #3b82f6;")
        hw_layout.addWidget(self.val_power, 1, 1, Qt.AlignmentFlag.AlignCenter)

        hw_layout.addWidget(QLabel("VRAM sử dụng"), 0, 2, Qt.AlignmentFlag.AlignCenter)
        self.val_vram = QLabel("-- / 4.0 GB")
        self.val_vram.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.val_vram.setStyleSheet("color: #e4e4e7;")
        hw_layout.addWidget(self.val_vram, 1, 2, Qt.AlignmentFlag.AlignCenter)

        hw_layout.addWidget(QLabel("Khung trùng bỏ qua"), 0, 3, Qt.AlignmentFlag.AlignCenter)
        self.val_dedup = QLabel("0 (0.0%)")
        self.val_dedup.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.val_dedup.setStyleSheet("color: #a855f7;")
        hw_layout.addWidget(self.val_dedup, 1, 3, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hw_box)

        # Progress Box
        prog_box = QWidget()
        prog_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px;")
        prog_layout = QVBoxLayout(prog_box)
        prog_layout.setContentsMargins(10, 10, 10, 10)
        prog_layout.setSpacing(6)

        stats_line = QHBoxLayout()
        self.prog_text = QLabel("Tiến độ: 0% (0 / 0 khung)")
        self.prog_speed = QLabel("Tốc độ: -- fps • Còn lại: --:--")
        self.prog_speed.setStyleSheet("color: #71717a; font-family: monospace;")
        stats_line.addWidget(self.prog_text)
        stats_line.addStretch()
        stats_line.addWidget(self.prog_speed)
        prog_layout.addLayout(stats_line)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        btn_line = QHBoxLayout()
        self.start_btn = QPushButton("Bắt đầu Upscale")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.clicked.connect(self._start_pipeline)
        btn_line.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Tạm dừng")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        btn_line.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("Hủy")
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_pipeline)
        btn_line.addWidget(self.cancel_btn)

        btn_line.addStretch()
        open_folder_btn = QPushButton("Mở thư mục")
        open_folder_btn.clicked.connect(self._open_output_folder)
        btn_line.addWidget(open_folder_btn)
        prog_layout.addLayout(btn_line)
        layout.addWidget(prog_box)

        return tab

    def _build_subtitle_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 10, 8, 8)

        # File box
        file_box = QWidget()
        file_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        file_layout = QGridLayout(file_box)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(8)

        file_layout.addWidget(QLabel("Video nguồn:"), 0, 0)
        self.sub_input_edit = QLineEdit()
        self.sub_input_edit.setPlaceholderText("Chọn video cần tạo phụ đề (.mp4, .mkv, .mov)...")
        file_layout.addWidget(self.sub_input_edit, 0, 1)
        browse_in_btn = QPushButton("Chọn...")
        browse_in_btn.clicked.connect(self._browse_sub_input)
        file_layout.addWidget(browse_in_btn, 0, 2)

        file_layout.addWidget(QLabel("Video ra (MP4):"), 1, 0)
        self.sub_output_edit = QLineEdit()
        self.sub_output_edit.setPlaceholderText("Đường dẫn lưu video có phụ đề...")
        file_layout.addWidget(self.sub_output_edit, 1, 1)
        browse_out_btn = QPushButton("Lưu tại...")
        browse_out_btn.clicked.connect(self._browse_sub_output)
        file_layout.addWidget(browse_out_btn, 1, 2)

        self.sub_meta_label = QLabel("Chưa chọn video")
        self.sub_meta_label.setStyleSheet("color: #71717a; font-family: monospace; font-size: 11px;")
        file_layout.addWidget(self.sub_meta_label, 2, 1, 1, 2)
        layout.addWidget(file_box)

        # Settings box
        settings_box = QWidget()
        settings_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        set_layout = QVBoxLayout(settings_box)
        set_layout.setContentsMargins(10, 10, 10, 10)
        set_layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(10)

        s_col = QVBoxLayout()
        s_col.addWidget(QLabel("Ngôn ngữ nguồn:"))
        self.sub_lang_combo = QComboBox()
        self.sub_lang_combo.addItems([
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
        ])
        s_col.addWidget(self.sub_lang_combo)
        row1.addLayout(s_col)

        w_col = QVBoxLayout()
        w_col.addWidget(QLabel("Mô hình Whisper ASR:"))
        self.sub_whisper_combo = QComboBox()
        self.sub_whisper_combo.addItems([
            "large-v3 (Độ chính xác cao nhất - Khuyên dùng)",
            "medium",
            "small",
            "base",
        ])
        w_col.addWidget(self.sub_whisper_combo)
        row1.addLayout(w_col)

        g_col = QVBoxLayout()
        g_col.addWidget(QLabel("Mô hình Dịch (Gemini):"))
        self.sub_gemini_combo = QComboBox()
        self.sub_gemini_combo.addItems([
            "gemini-3.1-flash-lite (Nhanh & Tối ưu chi phí)",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash",
        ])
        g_col.addWidget(self.sub_gemini_combo)
        row1.addLayout(g_col)
        set_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)

        k_col = QVBoxLayout()
        k_col.addWidget(QLabel("Khóa Gemini API (hoặc biến GEMINI_API_KEY):"))
        self.sub_api_key_edit = QLineEdit()
        self.sub_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.sub_api_key_edit.setPlaceholderText("Nhập khóa API (để trống nếu đã có biến môi trường)...")
        k_col.addWidget(self.sub_api_key_edit)
        row2.addLayout(k_col)

        f_col = QVBoxLayout()
        f_col.addWidget(QLabel("Phông chữ hiển thị:"))
        self.sub_font_edit = QLineEdit("Noto Sans")
        self.sub_font_edit.setPlaceholderText("Noto Sans / Arial / Segoe UI...")
        f_col.addWidget(self.sub_font_edit)
        row2.addLayout(f_col)
        set_layout.addLayout(row2)

        c_col = QVBoxLayout()
        c_col.addWidget(QLabel("Ghi chú ngữ cảnh dịch thuật (Xưng hô anh/em/tôi, bối cảnh phim):"))
        self.sub_context_edit = QLineEdit()
        self.sub_context_edit.setPlaceholderText("Ví dụ: Xưng hô anh/em thân mật; bối cảnh công nghệ thông tin...")
        c_col.addWidget(self.sub_context_edit)
        set_layout.addLayout(c_col)

        chk_row = QHBoxLayout()
        self.sub_burn_chk = QCheckBox("Ghép cứng phụ đề vào video (Burn-in MP4)")
        self.sub_burn_chk.setChecked(True)
        self.sub_bilingual_chk = QCheckBox("Hiển thị song ngữ (Gốc ở trên, dịch ở dưới)")
        self.sub_bilingual_chk.setChecked(False)
        chk_row.addWidget(self.sub_burn_chk)
        chk_row.addSpacing(20)
        chk_row.addWidget(self.sub_bilingual_chk)
        chk_row.addStretch()
        set_layout.addLayout(chk_row)
        layout.addWidget(settings_box)

        # Progress Box
        prog_box = QWidget()
        prog_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px;")
        prog_layout = QVBoxLayout(prog_box)
        prog_layout.setContentsMargins(10, 10, 10, 10)
        prog_layout.setSpacing(6)

        self.sub_prog_text = QLabel("Tiến độ: Sẵn sàng")
        prog_layout.addWidget(self.sub_prog_text)

        self.sub_progress_bar = QProgressBar()
        self.sub_progress_bar.setRange(0, 100)
        self.sub_progress_bar.setValue(0)
        prog_layout.addWidget(self.sub_progress_bar)

        btn_line = QHBoxLayout()
        self.sub_start_btn = QPushButton("Bắt đầu tạo Phụ đề")
        self.sub_start_btn.setObjectName("primaryBtn")
        self.sub_start_btn.clicked.connect(self._start_subtitle)
        btn_line.addWidget(self.sub_start_btn)

        self.sub_cancel_btn = QPushButton("Hủy")
        self.sub_cancel_btn.setObjectName("dangerBtn")
        self.sub_cancel_btn.setEnabled(False)
        self.sub_cancel_btn.clicked.connect(self._cancel_subtitle)
        btn_line.addWidget(self.sub_cancel_btn)

        btn_line.addStretch()
        open_sub_folder_btn = QPushButton("Mở thư mục")
        open_sub_folder_btn.clicked.connect(self._open_sub_folder)
        btn_line.addWidget(open_sub_folder_btn)
        prog_layout.addLayout(btn_line)
        layout.addWidget(prog_box)

        layout.addStretch()
        return tab

    def _build_dubbing_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 10, 8, 8)

        # 1. File Selection Panel
        file_box = QWidget()
        file_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        file_layout = QGridLayout(file_box)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(8)

        file_layout.addWidget(QLabel("Video vào:"), 0, 0)
        self.dub_input_edit = QLineEdit()
        self.dub_input_edit.setPlaceholderText("Chọn video nguồn cần lồng tiếng (.mp4, .mkv, .mov)...")
        file_layout.addWidget(self.dub_input_edit, 0, 1)
        browse_in_btn = QPushButton("Chọn...")
        browse_in_btn.clicked.connect(self._browse_dub_input)
        file_layout.addWidget(browse_in_btn, 0, 2)

        file_layout.addWidget(QLabel("Thư mục dự án:"), 1, 0)
        self.dub_project_edit = QLineEdit()
        self.dub_project_edit.setPlaceholderText("Thư mục lưu artifacts (mặc định: dub_project_<tên_video>)...")
        file_layout.addWidget(self.dub_project_edit, 1, 1)
        browse_out_btn = QPushButton("Chọn...")
        browse_out_btn.clicked.connect(self._browse_dub_project)
        file_layout.addWidget(browse_out_btn, 1, 2)

        self.dub_meta_label = QLabel("Chưa chọn video")
        self.dub_meta_label.setStyleSheet("color: #71717a; font-family: monospace; font-size: 11px;")
        file_layout.addWidget(self.dub_meta_label, 2, 1, 1, 2)
        layout.addWidget(file_box)

        # 2. Settings Panel
        settings_box = QWidget()
        settings_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        set_layout = QVBoxLayout(settings_box)
        set_layout.setContentsMargins(10, 10, 10, 10)
        set_layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(10)

        t_col = QVBoxLayout()
        t_col.addWidget(QLabel("Bộ tổng hợp giọng nói (TTS):"))
        self.dub_backend_combo = QComboBox()
        self.dub_backend_combo.addItems([
            "OpenVoice v2 (Clone âm sắc giọng gốc + F0)",
            "Google TTS (gTTS - Ổn định, tự nhiên, đa âm điệu)",
            "F5-TTS Vietnamese (Clone giọng mẫu - Yêu cầu GPU)",
            "Mock (Chạy thử nghiệm nhanh)",
        ])
        t_col.addWidget(self.dub_backend_combo)
        row1.addLayout(t_col)

        w_col = QVBoxLayout()
        w_col.addWidget(QLabel("Mô hình Whisper ASR:"))
        self.dub_whisper_combo = QComboBox()
        self.dub_whisper_combo.addItems([
            "large-v3 (Độ chính xác cao nhất - Khuyên dùng)",
            "medium",
            "small",
            "base",
        ])
        w_col.addWidget(self.dub_whisper_combo)
        row1.addLayout(w_col)

        g_col = QVBoxLayout()
        g_col.addWidget(QLabel("Mô hình Dịch thuật (Gemini):"))
        self.dub_gemini_combo = QComboBox()
        self.dub_gemini_combo.addItems([
            "gemini-2.5-flash (Khuyên dùng - Cân bằng & khớp số âm tiết)",
            "gemini-2.5-pro (Dịch sâu bối cảnh)",
            "gemini-2.0-flash",
        ])
        g_col.addWidget(self.dub_gemini_combo)
        row1.addLayout(g_col)
        set_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)

        spk_col = QVBoxLayout()
        spk_col.addWidget(QLabel("Số lượng nhân vật (Speakers):"))
        self.dub_spk_combo = QComboBox()
        self.dub_spk_combo.addItems([
            "Tự động phát hiện (SpeechBrain ECAPA-TDNN)",
            "1 nhân vật",
            "2 nhân vật",
            "3 nhân vật",
            "4 nhân vật",
        ])
        spk_col.addWidget(self.dub_spk_combo)
        row2.addLayout(spk_col)

        step_col = QVBoxLayout()
        step_col.addWidget(QLabel("Chạy đến bước (Target Step):"))
        self.dub_step_combo = QComboBox()
        self.dub_step_combo.addItems([
            "Toàn bộ quy trình (S0 -> S9)",
            "S1 - Tách Demucs",
            "S2 - Nhận dạng Faster-Whisper",
            "S3 - Phân cụm Diarization",
            "S4 - Trích xuất Voice Profile",
            "S5 - Dịch Gemini khớp thời lượng",
            "S6 - Tổng hợp TTS tiếng Việt",
            "S7 - Khớp thời lượng Snapping",
            "S8 - Trộn âm & Ducking",
        ])
        step_col.addWidget(self.dub_step_combo)
        row2.addLayout(step_col)

        k_col = QVBoxLayout()
        k_col.addWidget(QLabel("Gemini API Key (hoặc biến môi trường):"))
        self.dub_api_key_edit = QLineEdit()
        self.dub_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.dub_api_key_edit.setPlaceholderText("Nhập khóa API (để trống nếu đã có biến môi trường)...")
        k_col.addWidget(self.dub_api_key_edit)
        row2.addLayout(k_col)
        set_layout.addLayout(row2)

        c_col = QVBoxLayout()
        c_col.addWidget(QLabel("Ghi chú ngữ cảnh dịch thuật (Tính cách nhân vật, bối cảnh, xưng hô):"))
        self.dub_context_edit = QLineEdit()
        self.dub_context_edit.setPlaceholderText("Ví dụ: Peter Griffin nói hài hước mỉa mai; xưng hô thân mật Peter/Lois...")
        c_col.addWidget(self.dub_context_edit)
        set_layout.addLayout(c_col)
        layout.addWidget(settings_box)

        # 3. Progress Box
        prog_box = QWidget()
        prog_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px;")
        prog_layout = QVBoxLayout(prog_box)
        prog_layout.setContentsMargins(10, 10, 10, 10)
        prog_layout.setSpacing(6)

        self.dub_prog_text = QLabel("Tiến độ: Sẵn sàng")
        prog_layout.addWidget(self.dub_prog_text)

        self.dub_progress_bar = QProgressBar()
        self.dub_progress_bar.setRange(0, 100)
        self.dub_progress_bar.setValue(0)
        prog_layout.addWidget(self.dub_progress_bar)

        btn_line = QHBoxLayout()
        self.dub_start_btn = QPushButton("Bắt đầu Lồng tiếng")
        self.dub_start_btn.setObjectName("primaryBtn")
        self.dub_start_btn.clicked.connect(self._start_dubbing)
        btn_line.addWidget(self.dub_start_btn)

        self.dub_cancel_btn = QPushButton("Hủy")
        self.dub_cancel_btn.setObjectName("dangerBtn")
        self.dub_cancel_btn.setEnabled(False)
        self.dub_cancel_btn.clicked.connect(self._cancel_dubbing)
        btn_line.addWidget(self.dub_cancel_btn)

        btn_line.addStretch()
        open_dub_folder_btn = QPushButton("Mở thư mục dự án")
        open_dub_folder_btn.clicked.connect(self._open_dub_folder)
        btn_line.addWidget(open_dub_folder_btn)
        prog_layout.addLayout(btn_line)
        layout.addWidget(prog_box)

        layout.addStretch()
        return tab

    def _build_voice_clone_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setSpacing(8)
        root.setContentsMargins(8, 10, 8, 8)

        sub_tabs = QTabWidget()

        # ---------- Sub-tab 4.1: Clone voices from Video / Audio ----------
        st1 = QWidget()
        st1_lay = QVBoxLayout(st1)
        st1_lay.setSpacing(8)

        src_box = QWidget()
        src_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        src_grid = QGridLayout(src_box)
        src_grid.setSpacing(8)

        src_grid.addWidget(QLabel("Video/Audio nguồn:"), 0, 0)
        self.vc_src_edit = QLineEdit()
        self.vc_src_edit.setPlaceholderText("Chọn video hoặc file âm thanh chứa 1 hoặc nhiều nhân vật...")
        src_grid.addWidget(self.vc_src_edit, 0, 1, 1, 2)
        vc_src_btn = QPushButton("Chọn...")
        vc_src_btn.clicked.connect(self._vc_browse_source)
        src_grid.addWidget(vc_src_btn, 0, 3)

        src_grid.addWidget(QLabel("Tên bộ giọng (Label):"), 1, 0)
        self.vc_label_edit = QLineEdit()
        self.vc_label_edit.setPlaceholderText("Ví dụ: FamilyGuy_S1, NhanVatA...")
        src_grid.addWidget(self.vc_label_edit, 1, 1)

        src_grid.addWidget(QLabel("Số nhân vật:"), 1, 2)
        self.vc_num_spk_combo = QComboBox()
        self.vc_num_spk_combo.addItems(["Tự động (Auto)", "1", "2", "3", "4", "5", "6", "7", "8"])
        src_grid.addWidget(self.vc_num_spk_combo, 1, 3)

        src_grid.addWidget(QLabel("Whisper ASR:"), 2, 0)
        self.vc_whisper_combo = QComboBox()
        self.vc_whisper_combo.addItems(["medium", "base", "small", "large-v3"])
        src_grid.addWidget(self.vc_whisper_combo, 2, 1)

        src_grid.addWidget(QLabel("Ngưỡng tách:"), 2, 2)
        self.vc_thresh_spin = QDoubleSpinBox()
        self.vc_thresh_spin.setRange(0.45, 0.95)
        self.vc_thresh_spin.setSingleStep(0.02)
        self.vc_thresh_spin.setValue(0.72)
        src_grid.addWidget(self.vc_thresh_spin, 2, 3)

        src_grid.addWidget(QLabel("Cao độ nghe thử:"), 3, 0)
        self.vc_pitch_combo = QComboBox()
        self.vc_pitch_combo.addItems(["Tự động khớp cao độ F0", "Giữ nguyên gốc"])
        src_grid.addWidget(self.vc_pitch_combo, 3, 1, 1, 3)

        st1_lay.addWidget(src_box)

        ctrl_row = QHBoxLayout()
        self.vc_clone_btn = QPushButton("🧬 Phân tích & Xuất các bản Clone")
        self.vc_clone_btn.setObjectName("primaryBtn")
        self.vc_clone_btn.clicked.connect(self._vc_start_clone)
        ctrl_row.addWidget(self.vc_clone_btn)

        refresh_bank_btn = QPushButton("🔄 Tải lại Kho")
        refresh_bank_btn.clicked.connect(self._vc_refresh_bank_ui)
        ctrl_row.addWidget(refresh_bank_btn)

        play_sel_btn = QPushButton("▶ Nghe bản Clone (.mp3)")
        play_sel_btn.clicked.connect(self._vc_play_selected_clone)
        ctrl_row.addWidget(play_sel_btn)

        rename_sel_btn = QPushButton("✏️ Đổi tên")
        rename_sel_btn.clicked.connect(self._vc_rename_selected_clone)
        ctrl_row.addWidget(rename_sel_btn)

        del_sel_btn = QPushButton("🗑️ Xóa")
        del_sel_btn.setObjectName("dangerBtn")
        del_sel_btn.clicked.connect(self._vc_delete_selected_clone)
        ctrl_row.addWidget(del_sel_btn)
        st1_lay.addLayout(ctrl_row)

        self.vc_clone_prog = QProgressBar()
        self.vc_clone_prog.setValue(0)
        st1_lay.addWidget(self.vc_clone_prog)

        self.vc_bank_table = QTableWidget(0, 6)
        self.vc_bank_table.setHorizontalHeaderLabels(["Clone ID", "Tên hiển thị", "Vai gốc", "Thời lượng", "F0 (Hz)", "Thoại mẫu"])
        self.vc_bank_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.vc_bank_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.vc_bank_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        st1_lay.addWidget(self.vc_bank_table)

        sub_tabs.addTab(st1, "1. Clone giọng từ Video / Audio")

        # ---------- Sub-tab 4.2: Cross-video Speaker Analysis & Dubbing ----------
        st2 = QWidget()
        st2_lay = QVBoxLayout(st2)
        st2_lay.setSpacing(8)

        tgt_box = QWidget()
        tgt_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        tgt_grid = QGridLayout(tgt_box)
        tgt_grid.setSpacing(6)

        tgt_grid.addWidget(QLabel("Video đích cần lồng tiếng:"), 0, 0)
        self.vc_tgt_edit = QLineEdit()
        self.vc_tgt_edit.setPlaceholderText("Chọn Video thứ 2 cần phân tích nhân vật và lồng tiếng...")
        tgt_grid.addWidget(self.vc_tgt_edit, 0, 1, 1, 2)
        vc_tgt_btn = QPushButton("Chọn...")
        vc_tgt_btn.clicked.connect(self._vc_browse_target)
        tgt_grid.addWidget(vc_tgt_btn, 0, 3)

        tgt_grid.addWidget(QLabel("Số nhân vật đích:"), 1, 0)
        self.vc_tgt_num_combo = QComboBox()
        self.vc_tgt_num_combo.addItems(["Tự động (Auto)", "1", "2", "3", "4", "5", "6", "7", "8"])
        tgt_grid.addWidget(self.vc_tgt_num_combo, 1, 1)

        tgt_grid.addWidget(QLabel("Whisper ASR:"), 1, 2)
        self.vc_tgt_whisper_combo = QComboBox()
        self.vc_tgt_whisper_combo.addItems(["medium", "base", "small", "large-v3"])
        tgt_grid.addWidget(self.vc_tgt_whisper_combo, 1, 3)

        st2_lay.addWidget(tgt_box)

        ana_row = QHBoxLayout()
        self.vc_analyze_btn = QPushButton("🔍 Phân tích nhân vật trong Video đích")
        self.vc_analyze_btn.clicked.connect(self._vc_start_analyze_target)
        ana_row.addWidget(self.vc_analyze_btn)

        vc_f0_map_btn = QPushButton("🎯 Gợi ý theo F0")
        vc_f0_map_btn.clicked.connect(self._vc_apply_f0_mapping)
        ana_row.addWidget(vc_f0_map_btn)

        vc_ref_map_btn = QPushButton("🔄 Làm mới DS Clone")
        vc_ref_map_btn.clicked.connect(self._vc_rebuild_target_mapping_rows)
        ana_row.addWidget(vc_ref_map_btn)
        ana_row.addStretch()
        st2_lay.addLayout(ana_row)

        map_scroll = QScrollArea()
        map_scroll.setWidgetResizable(True)
        map_scroll.setMinimumHeight(150)
        self.vc_map_container = QWidget()
        self.vc_map_layout = QVBoxLayout(self.vc_map_container)
        self.vc_map_layout.setSpacing(6)
        self.vc_map_layout.addWidget(QLabel("Hãy bấm 'Phân tích nhân vật' để hiển thị danh sách nhân vật và ghép Clone."))
        self.vc_map_layout.addStretch()
        map_scroll.setWidget(self.vc_map_container)
        st2_lay.addWidget(map_scroll)

        cfg2_box = QWidget()
        cfg2_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        cfg2_grid = QGridLayout(cfg2_box)
        cfg2_grid.setSpacing(6)

        cfg2_grid.addWidget(QLabel("Gemini Model:"), 0, 0)
        self.vc_tgt_gemini_combo = QComboBox()
        self.vc_tgt_gemini_combo.addItems(["gemini-2.5-flash", "gemini-1.5-flash"])
        cfg2_grid.addWidget(self.vc_tgt_gemini_combo, 0, 1)

        cfg2_grid.addWidget(QLabel("Khớp cao độ:"), 0, 2)
        self.vc_tgt_pitch_combo = QComboBox()
        self.vc_tgt_pitch_combo.addItems(["Tự động khớp F0", "Giữ nguyên gốc", "Thủ công (semitones)"])
        cfg2_grid.addWidget(self.vc_tgt_pitch_combo, 0, 3)

        cfg2_grid.addWidget(QLabel("Dịch semitones:"), 1, 0)
        self.vc_tgt_pitch_spin = QDoubleSpinBox()
        self.vc_tgt_pitch_spin.setRange(-7.0, 7.0)
        self.vc_tgt_pitch_spin.setSingleStep(0.5)
        cfg2_grid.addWidget(self.vc_tgt_pitch_spin, 1, 1)

        cfg2_grid.addWidget(QLabel("Ngữ cảnh dịch:"), 1, 2)
        self.vc_tgt_context_edit = QLineEdit()
        self.vc_tgt_context_edit.setPlaceholderText("Ngữ cảnh phim, cách xưng hô...")
        cfg2_grid.addWidget(self.vc_tgt_context_edit, 1, 3)

        cfg2_grid.addWidget(QLabel("Gemini API Key:"), 2, 0)
        self.vc_tgt_api_key_edit = QLineEdit()
        self.vc_tgt_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.vc_tgt_api_key_edit.setPlaceholderText("Để trống nếu đã thiết lập GEMINI_API_KEY...")
        cfg2_grid.addWidget(self.vc_tgt_api_key_edit, 2, 1, 1, 3)

        st2_lay.addWidget(cfg2_box)

        self.vc_dub_prog = QProgressBar()
        self.vc_dub_prog.setValue(0)
        st2_lay.addWidget(self.vc_dub_prog)

        dub2_row = QHBoxLayout()
        self.vc_dub_start_btn = QPushButton("🎬 Bắt đầu Lồng tiếng theo Giọng đã gán")
        self.vc_dub_start_btn.setObjectName("primaryBtn")
        self.vc_dub_start_btn.clicked.connect(self._vc_start_dub_mapped)
        dub2_row.addWidget(self.vc_dub_start_btn)

        self.vc_dub_cancel_btn = QPushButton("Hủy")
        self.vc_dub_cancel_btn.setObjectName("dangerBtn")
        self.vc_dub_cancel_btn.setEnabled(False)
        self.vc_dub_cancel_btn.clicked.connect(self._vc_cancel_dub_mapped)
        dub2_row.addWidget(self.vc_dub_cancel_btn)
        dub2_row.addStretch()
        st2_lay.addLayout(dub2_row)

        sub_tabs.addTab(st2, "2. Lồng tiếng Video bằng Giọng Clone")

        # ---------- Sub-tab 4.3: Select Clone -> Read TTS -> Export .MP3 ----------
        st3 = QWidget()
        st3_lay = QVBoxLayout(st3)
        st3_lay.setSpacing(8)

        tts_box = QWidget()
        tts_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        tts_grid = QGridLayout(tts_box)
        tts_grid.setSpacing(8)

        tts_grid.addWidget(QLabel("Chọn Giọng Clone:"), 0, 0)
        self.vc_tts_clone_combo = QComboBox()
        tts_grid.addWidget(self.vc_tts_clone_combo, 0, 1, 1, 2)
        vc_tts_play_ref_btn = QPushButton("▶ Nghe mẫu")
        vc_tts_play_ref_btn.clicked.connect(self._vc_play_tts_ref)
        tts_grid.addWidget(vc_tts_play_ref_btn, 0, 3)

        tts_grid.addWidget(QLabel("Ngôn ngữ:"), 1, 0)
        self.vc_tts_lang_combo = QComboBox()
        self.vc_tts_lang_combo.addItems(["Tiếng Việt (vi)", "English (en)"])
        tts_grid.addWidget(self.vc_tts_lang_combo, 1, 1)

        tts_grid.addWidget(QLabel("Tốc độ đọc:"), 1, 2)
        self.vc_tts_speed_spin = QDoubleSpinBox()
        self.vc_tts_speed_spin.setRange(0.65, 1.55)
        self.vc_tts_speed_spin.setSingleStep(0.05)
        self.vc_tts_speed_spin.setValue(1.0)
        tts_grid.addWidget(self.vc_tts_speed_spin, 1, 3)

        tts_grid.addWidget(QLabel("Cao độ:"), 2, 0)
        self.vc_tts_pitch_combo = QComboBox()
        self.vc_tts_pitch_combo.addItems(["Tự động khớp F0", "Giữ nguyên gốc", "Thủ công (semitones)"])
        tts_grid.addWidget(self.vc_tts_pitch_combo, 2, 1)

        tts_grid.addWidget(QLabel("Dịch semitones:"), 2, 2)
        self.vc_tts_pitch_spin = QDoubleSpinBox()
        self.vc_tts_pitch_spin.setRange(-7.0, 7.0)
        self.vc_tts_pitch_spin.setSingleStep(0.5)
        tts_grid.addWidget(self.vc_tts_pitch_spin, 2, 3)

        tts_grid.addWidget(QLabel("Xuất file .MP3:"), 3, 0)
        self.vc_tts_out_edit = QLineEdit(str(Path("outputs/tts_clone_output.mp3")))
        tts_grid.addWidget(self.vc_tts_out_edit, 3, 1, 1, 2)
        vc_tts_save_btn = QPushButton("Chọn nơi lưu...")
        vc_tts_save_btn.clicked.connect(self._vc_browse_tts_out)
        tts_grid.addWidget(vc_tts_save_btn, 3, 3)

        st3_lay.addWidget(tts_box)

        st3_lay.addWidget(QLabel("Nội dung văn bản cần đọc:"))
        self.vc_tts_text_edit = QTextEdit()
        self.vc_tts_text_edit.setPlaceholderText("Nhập đoạn văn bản nhiều câu cần tổng hợp sang giọng clone và xuất .mp3...")
        self.vc_tts_text_edit.setMinimumHeight(110)
        st3_lay.addWidget(self.vc_tts_text_edit)

        self.vc_tts_prog = QProgressBar()
        self.vc_tts_prog.setValue(0)
        st3_lay.addWidget(self.vc_tts_prog)

        tts_btn_row = QHBoxLayout()
        self.vc_tts_start_btn = QPushButton("🔊 Đọc & Xuất file .MP3")
        self.vc_tts_start_btn.setObjectName("primaryBtn")
        self.vc_tts_start_btn.clicked.connect(self._vc_start_tts_mp3)
        tts_btn_row.addWidget(self.vc_tts_start_btn)

        self.vc_tts_play_out_btn = QPushButton("▶ Nghe file .MP3 vừa xuất")
        self.vc_tts_play_out_btn.clicked.connect(lambda: self._open_local_media(self.vc_tts_out_edit.text().strip()))
        tts_btn_row.addWidget(self.vc_tts_play_out_btn)
        tts_btn_row.addStretch()
        st3_lay.addLayout(tts_btn_row)

        sub_tabs.addTab(st3, "3. Đọc TTS & Xuất .MP3")

        root.addWidget(sub_tabs)
        self._vc_refresh_bank_ui()
        return tab

    def _init_hardware_info(self):
        """Read initial hardware telemetry."""
        gpu = check_gpu()
        if gpu["available"]:
            self.val_temp.setText(f"{gpu['temp_c']} °C")
            if gpu["power_w"] is not None:
                self.val_power.setText(f"{gpu['power_w']} W")
            total_gb = round(gpu["vram_total_mb"] / 1024, 1)
            self.val_vram.setText(f"0.0 / {total_gb} GB")
            self.log_box.append(f"[Phần cứng] Phát hiện {gpu['name']} ({total_gb} GB VRAM).")

    # ================= UPSCALE EVENT HANDLERS =================
    def _browse_input(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video Nguồn", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)"
        )
        if file:
            self.input_edit.setText(file)
            try:
                self.meta = get_video_metadata(file)
                self.meta_label.setText(
                    f"Gốc: {self.meta.width}x{self.meta.height} • {self.meta.fps:.2f} fps • Audio: {'Có' if self.meta.has_audio else 'Không'} • {self.meta.total_frames} khung"
                )
                p = Path(file)
                default_out = str(p.parent / f"{p.stem}_upscaled{p.suffix}")
                self.output_edit.setText(default_out)
            except Exception as e:
                self.meta_label.setText(f"Lỗi đọc metadata: {e}")

    def _browse_output(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "Lưu Video Đầu Ra", self.output_edit.text() or "", "MP4 Video (*.mp4)"
        )
        if file:
            self.output_edit.setText(file)

    def _start_pipeline(self):
        in_file = self.input_edit.text().strip()
        out_file = self.output_edit.text().strip()

        if not in_file or not os.path.exists(in_file):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video đầu vào hợp lệ.")
            return

        model_text = self.model_combo.currentText().lower()
        if "nomosuni" in model_text:
            model_name = "2xNomosUni_compact_otf_medium"
            model_file = "models/2xNomosUni_compact_otf_medium.onnx"
            model_scale = 2
        elif "span" in model_text:
            model_name = "2x-spanx2-ch48"
            model_file = "models/2x-spanx2-ch48.onnx"
            model_scale = 2
        elif "x2plus" in model_text:
            model_name = "2x-realesrgan-x2plus"
            model_file = "models/2x-realesrgan-x2plus.onnx"
            model_scale = 2
        elif "anime" in model_text:
            model_name = "realesr-animevideov3"
            model_file = "models/realesr-animevideov3.onnx"
            model_scale = 4
        else:
            model_name = "realesr-general-x4v3"
            model_file = "models/realesr-general-x4v3.onnx"
            model_scale = 4

        target_scale = 4 if "4x" in self.scale_combo.currentText() else 2

        tile_text = self.tile_combo.currentText()
        if "Toàn khung" in tile_text:
            tile_sz = 0
            tile_ov = 0
        elif "512" in tile_text:
            tile_sz = 512
            tile_ov = 16
        else:
            tile_sz = 256
            tile_ov = 16

        cfg = AppConfig(
            model=ModelConfig(
                name=model_name,
                path=model_file,
                scale=model_scale,
                target_scale=target_scale,
            ),
            inference=InferenceConfig(
                provider="auto",
                tile_size=tile_sz,
                tile_overlap=tile_ov,
            ),
            thermal=ThermalConfig(
                enabled=self.thermal_chk.isChecked(),
                temp_soft_c=self.temp_spin.value(),
                temp_hard_c=self.temp_spin.value() + 10,
                temp_resume_c=self.temp_spin.value() - 5,
            ),
            frame_saving=FrameSavingConfig(
                enable_dedup=self.dedup_chk.isChecked(),
            ),
            checkpoint=CheckpointConfig(
                enabled=self.checkpoint_chk.isChecked(),
                interval_frames=300,
            ),
            encode=EncodeConfig(
                codec="h264_nvenc" if "NVENC H.264" in self.encoder_combo.currentText() else "libx264",
                copy_audio=True,
            ),
            pre_process=PreProcessConfig(
                denoise=self.pre_denoise_spin.value() / 100.0,
            ),
            post_process=PostProcessConfig(
                recover_detail=self.recover_spin.value() / 100.0,
                sharpen=self.sharpen_spin.value() / 100.0,
                film_grain=0.015 if self.grain_chk.isChecked() else 0.0,
            ),
        )

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.status_badge.setText("Đang chạy")
        self.status_badge.setStyleSheet("color: #3b82f6; font-weight: 500;")

        self.worker = PipelineWorker(config=cfg, input_path=in_file, output_path=out_file)
        self.worker.progress_sig.connect(self._on_worker_progress)
        self.worker.log_sig.connect(self.log_box.append)
        self.worker.finished_sig.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_progress(self, frame_idx: int, total_frames: int, fps: float, eta_sec: float, metrics, dedup_rate: float):
        pct = int((frame_idx / max(1, total_frames)) * 100)
        self.progress_bar.setValue(pct)
        self.prog_text.setText(f"Tiến độ: {pct}% ({frame_idx} / {total_frames} khung)")

        mins = int(eta_sec // 60)
        secs = int(eta_sec % 60)
        self.prog_speed.setText(f"Tốc độ: {fps:.1f} fps • Còn lại: {mins:02d}:{secs:02d}")

        if metrics:
            self.val_temp.setText(f"{metrics.temp_c} °C")
            if metrics.temp_c >= self.temp_spin.value():
                self.val_temp.setStyleSheet("color: #f59e0b;")
            else:
                self.val_temp.setStyleSheet("color: #10b981;")
            if metrics.power_w:
                self.val_power.setText(f"{round(metrics.power_w, 1)} W")
            used_gb = round(metrics.vram_used_mb / 1024, 1)
            total_gb = round(metrics.vram_total_mb / 1024, 1)
            self.val_vram.setText(f"{used_gb} / {total_gb} GB")

        self.val_dedup.setText(f"{dedup_rate:.1f}%")

    def _on_worker_finished(self, success: bool, message: str):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.status_badge.setText("Sẵn sàng")
        self.status_badge.setStyleSheet("color: #10b981; font-weight: 500;")

        if success:
            self.log_box.append(f"[Xong] {message}")
            QMessageBox.information(self, "Thành công", message)
        else:
            self.log_box.append(f"[Lỗi] {message}")
            QMessageBox.critical(self, "Lỗi xử lý", message)

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.worker and self.worker.pipeline:
            self.worker.pipeline.set_paused(self.is_paused)

        if self.is_paused:
            self.pause_btn.setText("Tiếp tục")
            self.status_badge.setText("Tạm dừng")
            self.status_badge.setStyleSheet("color: #f59e0b;")
            self.log_box.append("[Hệ thống] Đã tạm dừng xử lý (GPU đang nghỉ).")
        else:
            self.pause_btn.setText("Tạm dừng")
            self.status_badge.setText("Đang chạy")
            self.status_badge.setStyleSheet("color: #3b82f6;")
            self.log_box.append("[Hệ thống] Tiếp tục xử lý...")

    def _cancel_pipeline(self):
        if self.worker:
            self.worker.cancel()
            self.log_box.append("[Hệ thống] Đang yêu cầu hủy xử lý...")

    def _open_output_folder(self):
        out_file = self.output_edit.text().strip()
        folder = Path(out_file).parent if out_file else Path.cwd()
        if folder.exists():
            os.startfile(folder)

    # ================= SUBTITLE EVENT HANDLERS =================
    def _browse_sub_input(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video Cần Tạo Phụ Đề", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)"
        )
        if file:
            self.sub_input_edit.setText(file)
            try:
                self.sub_meta = get_video_metadata(file)
                self.sub_meta_label.setText(
                    f"Gốc: {self.sub_meta.width}x{self.sub_meta.height} • Audio: {'Có' if self.sub_meta.has_audio else 'Không'} • {self.sub_meta.duration_sec:.1f}s"
                )
                p = Path(file)
                default_out = str(p.parent / f"{p.stem}_sub{p.suffix}")
                self.sub_output_edit.setText(default_out)
            except Exception as e:
                self.sub_meta_label.setText(f"Lỗi đọc metadata: {e}")

    def _browse_sub_output(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "Lưu Video Phụ Đề", self.sub_output_edit.text() or "", "MP4 Video (*.mp4)"
        )
        if file:
            self.sub_output_edit.setText(file)

    def _start_subtitle(self):
        in_file = self.sub_input_edit.text().strip()
        out_file = self.sub_output_edit.text().strip()

        if not in_file or not os.path.exists(in_file):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video đầu vào hợp lệ.")
            return

        api_key = self.sub_api_key_edit.text().strip()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        from ecoup.subtitle.config import SubtitleConfig

        cfg = SubtitleConfig()
        lang_text = self.sub_lang_combo.currentText()
        if "auto" in lang_text.lower():
            cfg.asr.language = None
        else:
            cfg.asr.language = lang_text.split()[0].strip()

        cfg.asr.model = self.sub_whisper_combo.currentText().split()[0].strip()
        cfg.translate.model = self.sub_gemini_combo.currentText().split()[0].strip()
        cfg.translate.context_note = self.sub_context_edit.text().strip()
        cfg.burn_in = self.sub_burn_chk.isChecked()
        cfg.burn.show_original = self.sub_bilingual_chk.isChecked()
        if self.sub_font_edit.text().strip():
            cfg.burn.font_name = self.sub_font_edit.text().strip()

        out_dir = str(Path(out_file).parent if out_file else Path(in_file).parent)
        cfg.translate.cache_dir = str(Path(out_dir) / "sub_cache")

        self.sub_start_btn.setEnabled(False)
        self.sub_cancel_btn.setEnabled(True)
        self.status_badge.setText("Đang tạo phụ đề")
        self.status_badge.setStyleSheet("color: #3b82f6; font-weight: 500;")

        self.sub_worker = SubtitleWorker(config=cfg, input_path=in_file, output_path=out_file, out_dir=out_dir)
        self.sub_worker.progress_sig.connect(self._on_sub_worker_progress)
        self.sub_worker.log_sig.connect(self.log_box.append)
        self.sub_worker.finished_sig.connect(self._on_sub_worker_finished)
        self.sub_worker.start()

    def _on_sub_worker_progress(self, step: str, frac: float):
        step_names = {
            "audio": ("Tách âm thanh", 15),
            "asr": ("Nhận dạng giọng nói Whisper", 45),
            "normalize": ("Chuẩn hóa mốc thời gian", 55),
            "translate": ("Dịch tiếng Việt Gemini", 80),
            "srt": ("Xuất tệp phụ đề SRT", 85),
            "burn": ("Ghép cứng video (FFmpeg libass)", 95),
        }
        name, pct = step_names.get(step, (step, int(frac * 100)))
        self.sub_progress_bar.setValue(pct)
        self.sub_prog_text.setText(f"Tiến độ ({name}): {pct}%")

    def _on_sub_worker_finished(self, success: bool, rep, message: str):
        self.sub_start_btn.setEnabled(True)
        self.sub_cancel_btn.setEnabled(False)
        self.status_badge.setText("Sẵn sàng")
        self.status_badge.setStyleSheet("color: #10b981; font-weight: 500;")
        self.sub_progress_bar.setValue(100 if success else 0)

        if success:
            self.log_box.append(f"[Phụ đề thành công] {message}")
            if rep and getattr(rep, "outputs", None):
                for k, v in rep.outputs.items():
                    self.log_box.append(f"  -> {k}: {v}")
            QMessageBox.information(self, "Thành công", message)
        else:
            self.log_box.append(f"[Phụ đề lỗi] {message}")
            QMessageBox.critical(self, "Lỗi tạo phụ đề", message)

    def _cancel_subtitle(self):
        if self.sub_worker:
            self.sub_worker.cancel()
            self.log_box.append("[Hệ thống] Đang yêu cầu hủy tiến trình phụ đề...")

    def _open_sub_folder(self):
        out_file = self.sub_output_edit.text().strip()
        folder = Path(out_file).parent if out_file else Path.cwd()
        if folder.exists():
            os.startfile(folder)

    # ================= DUBBING EVENT HANDLERS =================
    def _browse_dub_input(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video Cần Lồng Tiếng", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)"
        )
        if file:
            self.dub_input_edit.setText(file)
            try:
                self.dub_meta = get_video_metadata(file)
                self.dub_meta_label.setText(
                    f"Gốc: {self.dub_meta.width}x{self.dub_meta.height} • Audio: {'Có' if self.dub_meta.has_audio else 'Không'} • {self.dub_meta.duration_sec:.1f}s"
                )
                p = Path(file)
                default_proj = str(p.parent / f"dub_project_{p.stem}")
                self.dub_project_edit.setText(default_proj)
            except Exception as e:
                self.dub_meta_label.setText(f"Lỗi đọc metadata: {e}")

    def _browse_dub_project(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Chọn Thư Mục Dự Án Lồng Tiếng", self.dub_project_edit.text() or ""
        )
        if folder:
            self.dub_project_edit.setText(folder)

    def _start_dubbing(self):
        in_file = self.dub_input_edit.text().strip()
        proj_dir = self.dub_project_edit.text().strip()

        if not in_file or not os.path.exists(in_file):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video đầu vào hợp lệ.")
            return

        api_key = self.dub_api_key_edit.text().strip()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        from ecoup.dubbing.config import DubbingConfig

        if not proj_dir:
            p = Path(in_file)
            proj_dir = str(p.parent / f"dub_project_{p.stem}")

        cfg = DubbingConfig(project_dir=proj_dir)

        backend_text = self.dub_backend_combo.currentText().lower()
        if "openvoice" in backend_text or "clone âm sắc" in backend_text:
            cfg.tts.backend = "openvoice"
        elif "google" in backend_text or "gtts" in backend_text:
            cfg.tts.backend = "gtts"
        elif "f5" in backend_text:
            cfg.tts.backend = "f5"
        else:
            cfg.tts.backend = "mock"

        cfg.asr.model_size = self.dub_whisper_combo.currentText().split()[0].strip()
        cfg.translation.model = self.dub_gemini_combo.currentText().split()[0].strip()

        spk_text = self.dub_spk_combo.currentText()
        if "1" in spk_text:
            cfg.diarization.num_speakers = 1
        elif "2" in spk_text:
            cfg.diarization.num_speakers = 2
        elif "3" in spk_text:
            cfg.diarization.num_speakers = 3
        elif "4" in spk_text:
            cfg.diarization.num_speakers = 4
        else:
            cfg.diarization.num_speakers = None

        step_text = self.dub_step_combo.currentText()
        target_step = None
        if not step_text.startswith("Toàn bộ"):
            target_step = step_text.split()[0].strip()

        context_notes = self.dub_context_edit.text().strip()

        self.dub_start_btn.setEnabled(False)
        self.dub_cancel_btn.setEnabled(True)
        self.status_badge.setText("Đang lồng tiếng AI")
        self.status_badge.setStyleSheet("color: #3b82f6; font-weight: 500;")

        self.dub_worker = DubbingWorker(
            config=cfg,
            input_path=in_file,
            context_notes=context_notes,
            target_step=target_step,
        )
        self.dub_worker.progress_sig.connect(self._on_dub_worker_progress)
        self.dub_worker.log_sig.connect(self.log_box.append)
        self.dub_worker.finished_sig.connect(self._on_dub_worker_finished)
        self.dub_worker.start()

    def _on_dub_worker_progress(self, step: str, frac: float):
        step_map = {
            "S0": ("Tiếp nhận & kiểm tra âm thanh", 5),
            "S1": ("Tách nhạc nền & thoại (Demucs)", 20),
            "S2": ("Nhận dạng giọng nói (Faster-Whisper)", 40),
            "S3": ("Phân cụm nhân vật (SpeechBrain)", 55),
            "S4": ("Trích xuất mẫu giọng từng nhân vật", 65),
            "S5": ("Dịch thuật khớp thời lượng (Gemini)", 75),
            "S6": ("Tổng hợp giọng tiếng Việt (TTS)", 85),
            "S7": ("Khớp thời lượng từng câu (Snapping)", 92),
            "S8": ("Trộn âm thanh & Ducking nhạc nền", 96),
            "S9": ("Ghép âm thanh vào video (Stream copy)", 99),
        }
        desc, pct = step_map.get(step, (f"Đang xử lý {step}", int(frac * 100)))
        self.dub_progress_bar.setValue(pct)
        self.dub_prog_text.setText(f"Tiến độ ({desc}): {pct}%")

    def _on_dub_worker_finished(self, success: bool, res: dict, message: str):
        self.dub_start_btn.setEnabled(True)
        self.dub_cancel_btn.setEnabled(False)
        self.status_badge.setText("Sẵn sàng")
        self.status_badge.setStyleSheet("color: #10b981; font-weight: 500;")
        self.dub_progress_bar.setValue(100 if success else 0)

        if success:
            self.log_box.append(f"[Lồng tiếng thành công] {message}")
            if res.get("video_path"):
                self.log_box.append(f"  -> Video lồng tiếng: {res['video_path']}")
            if res.get("srt_path"):
                self.log_box.append(f"  -> Phụ đề SRT kèm theo: {res['srt_path']}")
            QMessageBox.information(self, "Thành công", f"{message}\n\nVideo: {res.get('video_path', '')}")
        else:
            self.log_box.append(f"[Lồng tiếng lỗi] {message}")
            QMessageBox.critical(self, "Lỗi lồng tiếng", message)

    def _cancel_dubbing(self):
        if self.dub_worker:
            self.dub_worker.cancel()
            self.log_box.append("[Hệ thống] Đang yêu cầu hủy tiến trình lồng tiếng...")

    def _open_dub_folder(self):
        proj_dir = self.dub_project_edit.text().strip()
        folder = Path(proj_dir) if proj_dir else Path.cwd()
        if folder.exists():
            os.startfile(folder)

    # ================= TAB 4: VOICE CLONING STUDIO HANDLERS =================
    def _open_local_media(self, path_str: str):
        p = Path(path_str) if path_str else None
        if not p or not p.exists():
            QMessageBox.warning(self, "Thông báo", "Chưa có tệp âm thanh để phát.")
            return
        if sys.platform.startswith("win"):
            os.startfile(str(p))
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(p)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(p)])

    @staticmethod
    def _vc_pitch_mode_from_text(text: str) -> str:
        low = (text or "").lower()
        if "giữ nguyên" in low or "none" in low:
            return "none"
        if "thủ công" in low or "manual" in low:
            return "manual"
        return "auto"

    def _vc_browse_source(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video/Audio nguồn", "", "Media Files (*.mp4 *.mkv *.avi *.mov *.webm *.wav *.mp3 *.m4a)"
        )
        if file:
            self.vc_src_edit.setText(file)

    def _vc_browse_target(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video đích cần lồng tiếng", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)"
        )
        if file:
            self.vc_tgt_edit.setText(file)

    def _vc_browse_tts_out(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "Lưu File Âm Thanh MP3", self.vc_tts_out_edit.text() or "outputs/tts_clone.mp3", "MP3 Audio (*.mp3)"
        )
        if file:
            self.vc_tts_out_edit.setText(file)

    def _vc_refresh_bank_ui(self):
        try:
            from ecoup.dubbing.voice_bank import VoiceBankManager
            bank = VoiceBankManager()
            voices = bank.list_voices()
        except Exception as e:
            self.log_box.append(f"[Voice Bank lỗi] {e}")
            return

        self.vc_bank_table.setRowCount(len(voices))
        for r, e in enumerate(voices):
            self.vc_bank_table.setItem(r, 0, QTableWidgetItem(e.clone_id))
            self.vc_bank_table.setItem(r, 1, QTableWidgetItem(e.display_name))
            self.vc_bank_table.setItem(r, 2, QTableWidgetItem(e.speaker_id))
            self.vc_bank_table.setItem(r, 3, QTableWidgetItem(f"{e.speaking_duration_s:.1f}s"))
            f0_s = f"{e.estimated_f0_hz:.0f} Hz" if e.estimated_f0_hz > 0 else "N/A"
            self.vc_bank_table.setItem(r, 4, QTableWidgetItem(f0_s))
            self.vc_bank_table.setItem(r, 5, QTableWidgetItem((e.reference_text or "")[:60]))

        cur_tts = self.vc_tts_clone_combo.currentText()
        self.vc_tts_clone_combo.clear()
        choices = bank.get_dropdown_choices()
        self.vc_tts_clone_combo.addItems(choices)
        if cur_tts in choices:
            self.vc_tts_clone_combo.setCurrentText(cur_tts)

        if self.vc_target_speakers:
            self._vc_rebuild_target_mapping_rows()

    def _vc_selected_clone_id(self) -> str:
        row = self.vc_bank_table.currentRow()
        if row < 0 and self.vc_bank_table.rowCount() > 0:
            row = 0
        if row < 0:
            return ""
        item = self.vc_bank_table.item(row, 0)
        return item.text().strip() if item else ""

    def _vc_play_selected_clone(self):
        cid = self._vc_selected_clone_id()
        if not cid:
            QMessageBox.information(self, "Thông báo", "Chưa có bản Clone nào được chọn.")
            return
        from ecoup.dubbing.voice_bank import VoiceBankManager
        entry = VoiceBankManager().get_voice(cid)
        if entry:
            self._open_local_media(entry.preview_mp3_path or entry.ref_wav_path)

    def _vc_rename_selected_clone(self):
        cid = self._vc_selected_clone_id()
        if not cid:
            return
        from PyQt6.QtWidgets import QInputDialog
        from ecoup.dubbing.voice_bank import VoiceBankManager
        bank = VoiceBankManager()
        entry = bank.get_voice(cid)
        new_name, ok = QInputDialog.getText(
            self, "Đổi tên Giọng Clone", "Tên hiển thị mới:", text=entry.display_name if entry else cid
        )
        if ok and new_name.strip():
            bank.rename_voice(cid, new_name.strip())
            self._vc_refresh_bank_ui()

    def _vc_delete_selected_clone(self):
        cid = self._vc_selected_clone_id()
        if not cid:
            return
        from ecoup.dubbing.voice_bank import VoiceBankManager
        if VoiceBankManager().delete_voice(cid):
            self.log_box.append(f"[Kho Giọng] Đã xóa bản clone '{cid}'.")
            self._vc_refresh_bank_ui()

    def _vc_start_clone(self):
        src = self.vc_src_edit.text().strip()
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Video/Audio nguồn hợp lệ.")
            return
        num_txt = self.vc_num_spk_combo.currentText()
        num_spk = int(num_txt) if num_txt.isdigit() else None

        self.vc_clone_btn.setEnabled(False)
        self.vc_clone_prog.setValue(0)
        self.status_badge.setText("Đang clone giọng...")
        self.vc_clone_worker = CloneVoicesWorker(
            source_path=src,
            label_prefix=self.vc_label_edit.text().strip(),
            num_speakers=num_spk,
            distance_threshold=float(self.vc_thresh_spin.value()),
            whisper_model=self.vc_whisper_combo.currentText(),
            pitch_match_mode=self._vc_pitch_mode_from_text(self.vc_pitch_combo.currentText()),
        )
        self.vc_clone_worker.progress_sig.connect(self._vc_on_clone_progress)
        self.vc_clone_worker.log_sig.connect(lambda msg: self.log_box.append(msg))
        self.vc_clone_worker.finished_sig.connect(self._vc_on_clone_finished)
        self.vc_clone_worker.start()

    def _vc_on_clone_progress(self, step: str, frac: float):
        base = {"S0": 5, "S1": 25, "S2": 45, "S3": 62, "S4": 75, "S4B_CLONE": 90}.get(step, 50)
        self.vc_clone_prog.setValue(min(99, int(base + frac * 10)))

    def _vc_on_clone_finished(self, success: bool, entries: list, message: str):
        self.vc_clone_btn.setEnabled(True)
        self.status_badge.setText("Sẵn sàng")
        self.vc_clone_prog.setValue(100 if success else 0)
        self._vc_refresh_bank_ui()
        if success:
            self.log_box.append(f"[Clone Giọng] {message}")
            QMessageBox.information(self, "Hoàn tất Clone Giọng", message)
        else:
            self.log_box.append(f"[Clone Giọng lỗi] {message}")
            QMessageBox.critical(self, "Lỗi Clone Giọng", message)

    def _vc_start_analyze_target(self):
        tgt = self.vc_tgt_edit.text().strip()
        if not tgt or not os.path.exists(tgt):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Video đích hợp lệ.")
            return
        num_txt = self.vc_tgt_num_combo.currentText()
        num_spk = int(num_txt) if num_txt.isdigit() else None

        self.vc_analyze_btn.setEnabled(False)
        self.status_badge.setText("Đang phân tích Video đích...")
        self.vc_analyze_worker = AnalyzeTargetSpeakersWorker(
            target_video=tgt,
            num_speakers=num_spk,
            whisper_model=self.vc_tgt_whisper_combo.currentText(),
        )
        self.vc_analyze_worker.log_sig.connect(lambda msg: self.log_box.append(msg))
        self.vc_analyze_worker.finished_sig.connect(self._vc_on_analyze_finished)
        self.vc_analyze_worker.start()

    def _vc_on_analyze_finished(self, success: bool, speakers: list, message: str):
        self.vc_analyze_btn.setEnabled(True)
        self.status_badge.setText("Sẵn sàng")
        if success:
            self.vc_target_speakers = speakers
            self._vc_rebuild_target_mapping_rows()
            self.log_box.append(f"[Phân tích Video đích] {message}")
        else:
            QMessageBox.critical(self, "Lỗi phân tích", message)

    def _vc_rebuild_target_mapping_rows(self, custom_mapping: Optional[dict[str, str]] = None):
        while self.vc_map_layout.count():
            item = self.vc_map_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.vc_target_combos = {}

        if not self.vc_target_speakers:
            self.vc_map_layout.addWidget(QLabel("Chưa có dữ liệu nhân vật đích."))
            self.vc_map_layout.addStretch()
            return

        from ecoup.dubbing.voice_bank import VoiceBankManager, suggest_speaker_mapping
        bank = VoiceBankManager()
        voices = bank.list_voices()
        special_opts = [
            "Tự clone từ nhân vật gốc [__self__]",
            "Giữ nguyên thoại gốc [__original__]",
        ]
        choices = bank.get_dropdown_choices() + special_opts
        label_by_id = {v.clone_id: bank.format_choice_label(v) for v in voices}
        label_by_id["__self__"] = special_opts[0]
        label_by_id["__original__"] = special_opts[1]

        mapping = custom_mapping if isinstance(custom_mapping, dict) else suggest_speaker_mapping(
            self.vc_target_speakers, voices, mode="order"
        )

        for spk in self.vc_target_speakers:
            spk_id = str(spk["speaker_id"])
            row_w = QWidget()
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(2, 2, 2, 2)

            f0_s = f"~{spk['estimated_f0_hz']:.0f}Hz" if spk.get("estimated_f0_hz", 0) > 0 else "N/A"
            lbl = QLabel(f"{spk['label']} ({spk['speaking_duration_s']:.1f}s, {f0_s})")
            lbl.setMinimumWidth(210)
            row_lay.addWidget(lbl)

            play_btn = QPushButton("▶ Nghe")
            prev_wav = spk.get("preview_wav", "")
            play_btn.clicked.connect(lambda _chk=False, p=prev_wav: self._open_local_media(p))
            row_lay.addWidget(play_btn)

            combo = QComboBox()
            combo.addItems(choices)
            chosen_id = mapping.get(spk_id, "__self__")
            chosen_lbl = label_by_id.get(chosen_id)
            if chosen_lbl and chosen_lbl in choices:
                combo.setCurrentText(chosen_lbl)
            row_lay.addWidget(combo, 1)

            self.vc_target_combos[spk_id] = combo
            self.vc_map_layout.addWidget(row_w)

        self.vc_map_layout.addStretch()

    def _vc_apply_f0_mapping(self):
        if not self.vc_target_speakers:
            return
        from ecoup.dubbing.voice_bank import VoiceBankManager, suggest_speaker_mapping
        voices = VoiceBankManager().list_voices()
        f0_map = suggest_speaker_mapping(self.vc_target_speakers, voices, mode="f0")
        self._vc_rebuild_target_mapping_rows(custom_mapping=f0_map)

    def _vc_start_dub_mapped(self):
        tgt = self.vc_tgt_edit.text().strip()
        if not tgt or not os.path.exists(tgt):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Video đích hợp lệ.")
            return

        api_key = self.vc_tgt_api_key_edit.text().strip()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        from ecoup.dubbing.config import DubbingConfig
        from ecoup.dubbing.voice_bank import VoiceBankManager

        p = Path(tgt)
        proj_dir = p.parent / f"dub_project_{p.stem}"
        bank = VoiceBankManager()

        spk_map: dict[str, str] = {}
        for spk_id, combo in self.vc_target_combos.items():
            norm_id = VoiceBankManager._normalize_choice_id(combo.currentText())
            if norm_id:
                spk_map[spk_id] = norm_id

        cfg = DubbingConfig(project_dir=str(proj_dir))
        cfg.tts.backend = "openvoice"
        cfg.tts.voice_bank_dir = str(bank.bank_dir)
        cfg.tts.speaker_voice_map = spk_map
        cfg.tts.pitch_match_mode = self._vc_pitch_mode_from_text(self.vc_tgt_pitch_combo.currentText())
        cfg.tts.pitch_shift_semitones = float(self.vc_tgt_pitch_spin.value())
        cfg.asr.model_size = self.vc_tgt_whisper_combo.currentText()
        cfg.translation.model = self.vc_tgt_gemini_combo.currentText()

        num_txt = self.vc_tgt_num_combo.currentText()
        cfg.diarization.num_speakers = int(num_txt) if num_txt.isdigit() else None

        self.vc_dub_start_btn.setEnabled(False)
        self.vc_dub_cancel_btn.setEnabled(True)
        self.vc_dub_prog.setValue(0)
        self.status_badge.setText("Đang lồng tiếng Clone...")

        self.vc_dub_worker = DubbingWorker(cfg, tgt, self.vc_tgt_context_edit.text().strip())
        self.vc_dub_worker.progress_sig.connect(self._vc_on_dub_mapped_progress)
        self.vc_dub_worker.log_sig.connect(lambda msg: self.log_box.append(msg))
        self.vc_dub_worker.finished_sig.connect(self._vc_on_dub_mapped_finished)
        self.vc_dub_worker.start()

    def _vc_on_dub_mapped_progress(self, step: str, frac: float):
        base = {"S0": 5, "S1": 15, "S2": 30, "S3": 42, "S4": 50, "S5": 62, "S6": 78, "S7": 88, "S8": 94, "S9": 98}.get(step, 50)
        self.vc_dub_prog.setValue(min(99, int(base + frac * 5)))

    def _vc_on_dub_mapped_finished(self, success: bool, res: dict, message: str):
        self.vc_dub_start_btn.setEnabled(True)
        self.vc_dub_cancel_btn.setEnabled(False)
        self.status_badge.setText("Sẵn sàng")
        self.vc_dub_prog.setValue(100 if success else 0)
        if success:
            self.log_box.append(f"[Studio Lồng tiếng] Thành công: {res.get('video_path', '')}")
            QMessageBox.information(self, "Hoàn tất", f"{message}\n\nVideo: {res.get('video_path', '')}")
        else:
            self.log_box.append(f"[Studio Lồng tiếng lỗi] {message}")
            QMessageBox.critical(self, "Lỗi", message)

    def _vc_cancel_dub_mapped(self):
        if self.vc_dub_worker:
            self.vc_dub_worker.cancel()

    def _vc_play_tts_ref(self):
        choice = self.vc_tts_clone_combo.currentText().strip()
        if not choice:
            QMessageBox.information(self, "Thông báo", "Chưa có bản Giọng Clone nào để nghe thử.")
            return
        from ecoup.dubbing.voice_bank import VoiceBankManager
        entry = VoiceBankManager().get_voice(choice)
        if entry:
            self._open_local_media(entry.preview_mp3_path or entry.ref_wav_path)

    def _vc_start_tts_mp3(self):
        choice = self.vc_tts_clone_combo.currentText().strip()
        text = self.vc_tts_text_edit.toPlainText().strip()
        out_mp3 = self.vc_tts_out_edit.text().strip()
        if not choice:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một bản Giọng Clone trước.")
            return
        if not text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập nội dung văn bản cần đọc.")
            return
        if not out_mp3:
            out_mp3 = str(Path("outputs/tts_clone_output.mp3"))
            self.vc_tts_out_edit.setText(out_mp3)

        lang = "en" if "(en)" in self.vc_tts_lang_combo.currentText().lower() else "vi"
        self.vc_tts_start_btn.setEnabled(False)
        self.vc_tts_prog.setValue(0)
        self.status_badge.setText("Đang xuất MP3...")
        self.vc_tts_worker = TtsCloneMp3Worker(
            clone_spec=choice,
            text=text,
            output_mp3=out_mp3,
            base_lang=lang,
            speed=float(self.vc_tts_speed_spin.value()),
            pitch_match_mode=self._vc_pitch_mode_from_text(self.vc_tts_pitch_combo.currentText()),
            pitch_shift_semitones=float(self.vc_tts_pitch_spin.value()),
        )
        self.vc_tts_worker.progress_sig.connect(lambda f: self.vc_tts_prog.setValue(int(f * 100)))
        self.vc_tts_worker.log_sig.connect(lambda msg: self.log_box.append(msg))
        self.vc_tts_worker.finished_sig.connect(self._vc_on_tts_finished)
        self.vc_tts_worker.start()

    def _vc_on_tts_finished(self, success: bool, rep: dict, message: str):
        self.vc_tts_start_btn.setEnabled(True)
        self.status_badge.setText("Sẵn sàng")
        self.vc_tts_prog.setValue(100 if success else 0)
        if success:
            self.log_box.append(f"[TTS Xuất MP3] {rep.get('mp3_path', '')} ({rep.get('duration_s', 0)}s)")
            QMessageBox.information(self, "Xuất MP3 thành công", f"{message}\nFile: {rep.get('mp3_path', '')}")
        else:
            QMessageBox.critical(self, "Lỗi xuất MP3", message)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
