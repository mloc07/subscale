# Ghi chép khảo sát ứng viên Module Lồng tiếng (Phase 0)

> Ngày kiểm tra: 07/11/2026. Môi trường kiểm thử: Google Colab GPU NVIDIA L4 (24GB VRAM), CUDA 13.0, PyTorch 2.14.1+cu130, Python 3.13.16.

---

## 1. Bảng tổng hợp ứng viên đã xác minh thực tế

| Nhóm công việc | Ứng viên / Mô hình | Giấy phép | Yêu cầu VRAM | Tốc độ thực tế (L4) | Hỗ trợ tiếng Việt | Trạng thái xác minh |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tách âm (Stem Separation)** | `demucs` (`htdemucs` 2-stems) | MIT | ~2.5 GB đỉnh | 12.48s cho clip 68.8s (~5.5x realtime) | N/A (đa âm thanh) | **Đạt xuất sắc**: Tách sạch thoại (`vocals.wav`) và nhạc nền (`no_vocals.wav`) 44.1kHz stereo. |
| **Nhận dạng giọng nói (ASR)** | `faster-whisper` (`base` / `small`) | MIT | < 1.0 GB | 3.31s cho clip 68.8s (~20x realtime) | Hỗ trợ 99+ ngôn ngữ | **Đạt xuất sắc**: Nhận diện chuẩn xác 100% lời thoại tiếng Anh của Family Guy, cần preload CUDA 12 libs. |
| **Dịch thuật có ngân sách** | `gemini-2.5-flash` | API Google | 0 GB (Cloud) | ~8.37s cho 8 câu thoại | Tự nhiên, tiếng Việt bản xứ | **Đạt xuất sắc**: Prompt khống chế số âm tiết (`max_syllables_budget`) giúp bản dịch vừa vặn với độ dài khung hình. |
| **Speaker Embedding** | `speechbrain/spkrec-ecapa-voxceleb` | Apache 2.0 | < 1.2 GB | 7.43s nạp model, < 0.1s/câu | N/A (trích xuất đặc trưng âm học) | **Đạt xuất sắc**: Không cần HF Token; vector 192 chiều, Cosine similarity giữa 2 đoạn cùng nhân vật đạt **0.6074**. |
| **TTS Zero-shot Clone** | `danhtran2mind/Vi-F5-TTS` | MIT | 5.71 GB đỉnh | 3.40s cho câu thoại 4.26s | Rất tốt với ref tiếng Việt | **Đạt điều kiện**: Chạy tốt trên L4/T4; khi cross-lingual (ref tiếng Anh -> sinh tiếng Việt) cần xử lý tokenizer hoặc kết hợp hybrid. |
| **TTS Tiếng Việt chuẩn** | `gTTS` (Google Translate TTS) | MIT | 0 GB | < 0.5s/câu | Phát âm tiếng Việt chuẩn 100% | **Đạt**: Hoạt động ổn định trên Colab, thích hợp làm fallback backend. |

---

## 2. Các phát hiện kỹ thuật quan trọng trong Phase 0

1. **Xung đột NumPy & Numba trên Python 3.13:**
   - NumPy 2.5 mặc định trên Colab gây lỗi với `numba`. Cần ghim `numpy<2.4` (hoặc `numpy~=2.3.5`).
2. **CUDA 12 vs CUDA 13 trên Colab:**
   - Hệ thống Colab dùng CUDA 13. Cần preload các thư viện `nvidia-cublas-cu12` và `nvidia-cudnn-cu12` (như đã có trong `src/ecoup/subtitle/asr.py`) để `ctranslate2` và `faster-whisper` chạy trên GPU.
3. **Ngân sách thời lượng dịch thuật (Duration Budget):**
   - Công thức tính: `max_syllables_budget = int(duration_sec * 3.8)` cho kết quả câu dịch tiếng Việt có độ dài ăn khớp hoàn hảo với thời lượng nói của nhân vật gốc.
4. **Không cần token Hugging Face cho Diarization:**
   - `speechbrain/spkrec-ecapa-voxceleb` là giải pháp thay thế hoàn hảo cho `pyannote`, không bắt người dùng phải đăng ký tài khoản Hugging Face hay tạo token bí mật.
