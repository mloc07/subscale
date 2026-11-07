# Eco Video Upscaler (`ecoup`)

Hệ thống AI Super-Resolution cho Video tiết kiệm VRAM và kiểm soát nhiệt độ GPU thông minh, tối ưu hóa cho cả máy tính xách tay cá nhân (như NVIDIA GeForce RTX 3050 Laptop / 4GB VRAM) lẫn môi trường điện toán đám mây Google Colab (Tesla T4 16GB).

---

## 1. Điểm nổi bật & Tính năng mới

- **Kiến trúc Video Super-Resolution (SeedVR2-3B DiT)**: Tích hợp mô hình Diffusion Transformer chuyên phục chế video của ByteDance (ICLR 2026), khử triệt để hiện tượng nhấp nháy ánh sáng (temporal flicker), tái tạo chi tiết da người chân thực (sợi tóc, bờ mi, khóe môi, vân da) mà không bị "bệt/nhựa".
- **Hệ thống Clip Chunking & Cross-Fade Blending**: Xử lý video theo từng cụm thời gian (`clip_length: 16`, `clip_overlap: 2`) kết hợp hòa trộn tuyến tính (cross-fade) tại vùng biên gối đầu, đảm bảo chuyển động liền mạch và bảo toàn 100% số lượng khung hình gốc.
- **Bộ Preset Tối ưu Chân dung & Nhận dạng**:
  - *Denoise (0.10)*: Khử viền răng cưa và artifact nén 720p trước khi đưa vào mạng AI.
  - *Wavelet Color Fix*: Triệt tiêu sai lệch màu sắc (Delta-E giảm từ 3.91 xuống 3.37) và ổn định quang độ.
  - *Identity Detail Recovery (0.10)*: Hòa trộn 10% ảnh gốc nhằm bảo toàn tuyệt đối giải phẫu khuôn mặt và nụ cười.
  - *Unsharp Masking (0.15)*: Tách bạch chân thực từng sợi tóc và lông mi.
- **Tiết kiệm VRAM (Tiling & Overlap Blending)**: Hỗ trợ xử lý video 1080p, 1440p, 4K trên GPU 4GB VRAM mà không lo bị tràn bộ nhớ (Out-Of-Memory) nhờ chia mảnh thông minh (PSNR > 40 dB).
- **Thermal Governor (Kiểm soát nhiệt tự động)**: Tự động điều tiết tốc độ (dynamic micro-sleeps) khi GPU chạm ngưỡng nhiệt mong muốn (mặc định 70°C), tự động tạm dừng an toàn nếu quá nhiệt và khôi phục khi máy nguội.
- **Checkpoint & Resume theo phân đoạn**: Tự động lưu tiến độ định kỳ (mặc định mỗi 300 khung) vào file `state.json`. Khi bị ngắt quãng (hoặc Colab timeout), hệ thống tiếp tục xử lý chính xác từ khung gián đoạn mà không làm mất thời gian hay lệch tiếng.
- **NVENC Hardware Acceleration**: Tận dụng nhân mã hóa phần cứng của card đồ họa NVIDIA để xuất video thành phẩm tốc độ cao.
- **Tạo & Gắn phụ đề tự động (Subtitle Pipeline - Whisper + Gemini)**: Tự động trích xuất âm thanh, nhận diện tiếng nói đa ngôn ngữ siêu tốc bằng `faster-whisper` (hỗ trợ `large-v3`, `medium`, `small`, `base` trên CUDA/CPU), dịch thuật ngữ cảnh tự nhiên bằng Google Gemini (`gemini-2.5-flash` / `gemini-3.1-flash-lite`), tự động ngắt dòng thông minh (CPL ≤ 42, CPS ≤ 17), xuất phụ đề chuẩn `.srt` / `.ass` (kèm Noto Sans/Arial styling) và burn-in trực tiếp vào video bằng FFmpeg `libass`. Tự động giải phóng 100% VRAM ngay sau ASR để không xung đột với mô hình upscale.
- **Lồng tiếng Đa nhân vật Tự động (AI Multi-Speaker Dubbing Studio - S0 -> S9)**:
  - Tách vocal và nhạc nền chuẩn 44.1kHz bằng Demucs (`htdemucs`).
  - Nhận diện lời thoại đa ngôn ngữ bằng Faster-Whisper (Large-v3/Medium/Small) kèm lọc ảo giác (hallucination filter).
  - Phân cụm và nhận diện giọng nhân vật tự động (Speaker Diarization) bằng SpeechBrain ECAPA-TDNN 192-dim không yêu cầu token HuggingFace, tự động chấm điểm độ tinh sạch giọng thoại (SNR + Flatness).
  - Dịch thuật kiểm soát ngân sách thời lượng âm tiết (Syllable-Budget Aware Translation) bằng Google Gemini (`gemini-2.5-flash`), tự động viết lại nếu vượt thời lượng thoại gốc (ngưỡng 3.8 âm tiết/giây).
  - Tổng hợp giọng nói tiếng Việt đa dạng nhân vật bằng `gTTS` kèm Pitch-shift timbral warping hoặc zero-shot voice cloning bằng `F5-TTS` (`danhtran2mind/Vi-F5-TTS`).
  - Tự động co giãn thời lượng chuẩn (atempo snapping $\le$ 1.25x).
  - Hòa âm thông minh: BGM Ducking -8dB với đường cong chuyển tiếp 50ms mượt mà, chuẩn hóa âm lượng và Peak Limiter chống rè (0.95 max).
  - Xuất video chuẩn Stream Copy (`-c:v copy`) không nén lại hình ảnh, không chèn watermark hay metadata cưỡng bức, xuất kèm file phụ đề SRT song hành.

---

## 2. Hướng dẫn chọn Mô hình AI (Model Selection)

| Mô hình | Môi trường tối ưu | Ưu điểm nổi bật | Khi nào nên dùng? |
|---|---|---|---|
| 👑 **SeedVR2-3B (DiT)** | Google Colab (Tesla T4 16GB / L4 24GB / A100) | Triệt tiêu flicker, da người cực nét tự nhiên, giữ trọn nhận dạng | **Khuyên dùng số 1** cho video người thật, chân dung, vlog, podcast từ 720p lên 1440p |
| ⚡ **NomosUni 2x / RealESRGAN** | Cục bộ (RTX 3050 Laptop 4GB) | VRAM chỉ ~1.2 – 1.8 GB, GPU cực mát (<70°C), tốc độ cao | Phù hợp xem trước (local preview) trên máy cá nhân hoặc video hoạt hình/anime |
| ⚙️ **CCSR v2 (Diffusion)** | Google Colab (Tesla T4 16GB) | Kiểm soát cấu trúc qua ControlNet (ghim `diffusers==0.21.4`) | Dự phòng khi muốn can thiệp sâu vào cấu trúc hình ảnh qua prompt/control strength |

### Khuyến nghị cấu hình phần cứng cho SeedVR2 (Hardware Recommendations)

| GPU | VRAM khả dụng | `memory_mode` | `precision` | `sub_len` | `model_resolution` | Tốc độ đo đạc thực tế |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NVIDIA Tesla T4** | 15.0 GB | `offload` | `fp16` / `bf16` | 5 | `native` | ~0.14 – 0.18 FPS (VRAM 9.85 GB) |
| **NVIDIA L4** | 22.0 GB | **`resident`** | **`fp16`** | 5 | `native` | **~0.35 – 0.51 FPS (VRAM 16.19 GB)** |
| **NVIDIA A100 (40GB/80GB)** | 40.0 / 80.0 GB | `resident` | `fp16` | 9 hoặc 13 | `native` hoặc `match_target` | **> 0.60 FPS** |

> [!TIP]
> **Chế độ Resident (`memory_mode: resident`)** loại bỏ hoàn toàn quá trình chuyển đổi DiT và VAE qua lại giữa RAM CPU và GPU qua PCIe trong mỗi khối, giúp tăng tốc **2.4 lần** (từ 69s xuống 28s cho 10 khung hình). Chế độ này tự động kích hoạt trên các GPU có từ 14GB VRAM trống trở lên (L4, A100, RTX 3090/4090).

---

## 3. Cài đặt hệ thống

### Yêu cầu tiên quyết
- Hệ điều hành: Windows 10/11 hoặc Linux (Ubuntu 22.04+).
- GPU: NVIDIA GPU hỗ trợ CUDA (RTX 3050 trở lên cho máy cục bộ; Tesla T4 trở lên cho Colab).
- Python: 3.10 – 3.11.
- FFmpeg: Đã được cài đặt và có trong PATH (`winget install Gyan.FFmpeg` trên Windows).

### Cài đặt môi trường
```powershell
# Kích hoạt virtualenv
.\.venv\Scripts\Activate.ps1

# Cài đặt gói ecoup cơ bản (cho upscale)
pip install -e .

# Hoặc cài đặt đầy đủ kèm tính năng Phụ đề tự động (faster-whisper, google-genai, pysubs2...)
pip install -e ".[subtitle]"

# Cài đặt đầy đủ tính năng Lồng tiếng Đa nhân vật (Demucs, SpeechBrain, gTTS...)
pip install -e ".[dubbing]"
```

### Thiết lập API Key cho Dịch thuật (Gemini)
Để sử dụng tính năng dịch phụ đề và lồng tiếng tự động bằng Google Gemini, thiết lập biến môi trường `GEMINI_API_KEY`:
```powershell
# Trên Windows PowerShell
$env:GEMINI_API_KEY="AIzaSy..."

# Trên Linux / macOS / Google Colab
export GEMINI_API_KEY="AIzaSy..."
```

### Kiểm tra chẩn đoán hệ thống
```powershell
python -m ecoup.cli doctor
```

---

## 4. Hướng dẫn sử dụng

### 4.1. Giao diện WebUI trên Google Colab / Cloud
Chạy ứng dụng Gradio tương thích hoàn toàn với Google Colab, hỗ trợ tạo liên kết công khai `gradio.live`:
```bash
python colab_app.py --share
```
*Giao diện tích hợp 3 tab chuyên biệt:*
- **⚡ Phục chế & Siêu phân giải (Upscale AI)**: Đầy đủ các thanh trượt điều chỉnh: Denoise, Detail Recovery, Sharpen, Film Grain, lựa chọn mô hình SeedVR2/CCSR/NomosUni, và nút Hủy an toàn.
- **📝 Phụ đề tự động (Whisper + Gemini)**: Nhận diện tiếng nói tự động, dịch ngữ cảnh sang tiếng Việt (hoặc ngôn ngữ tùy chọn), xuất file `.srt`, `.ass` và video đã burn-in phụ đề chuẩn Noto Sans.
- **🎙️ Lồng tiếng AI (Multi-Speaker Dubbing Studio)**: Tự động tách thoại & nhạc nền (Demucs), phân vai nhân vật (SpeechBrain), dịch khớp thời lượng (Gemini 2.5 Flash), tổng hợp giọng đọc tiếng Việt theo nhân vật (gTTS / F5-TTS) và hòa âm Ducking BGM chuyên nghiệp.

### 4.2. Giao diện Desktop (PyQt6)
Dành cho máy tính cá nhân cục bộ, tích hợp cả 3 tab Upscale, Subtitle và Dubbing với luồng worker nền không gây đơ UI:
```powershell
python -m gui.main_window
```

### 4.3. Dòng lệnh (CLI)

#### Phục chế & Siêu phân giải (Upscale)
```powershell
# Upscale video cơ bản (mặc định 2x)
python -m ecoup.cli run -i input.mp4 -o output.mp4

# Upscale với tỷ lệ tùy chỉnh (ví dụ 4x)
python -m ecoup.cli run -i input.mp4 -o output.mp4 --scale 4

# Tiếp tục tác vụ bị gián đoạn từ checkpoint
python -m ecoup.cli run -i input.mp4 -o output.mp4 --resume
```

#### Tạo & Gắn phụ đề tự động (Subtitle)
```powershell
# Nhận diện tiếng nói, dịch sang tiếng Việt và burn-in vào video (mặc định)
python -m ecoup.cli subtitle -i video.mp4 -o video_sub.mp4 --target-lang vi --burn

# Chỉ nhận diện và xuất file SRT gốc (bỏ qua dịch thuật và burn-in)
python -m ecoup.cli subtitle -i video.mp4 --no-translate --no-burn

# Tùy chọn model Whisper (medium/large-v3) và font chữ
python -m ecoup.cli subtitle -i video.mp4 -o video_sub.mp4 --whisper-model medium --font-name "Noto Sans" --font-size 22
```

#### Lồng tiếng Đa nhân vật Tự động (Dubbing)
```powershell
# Lồng tiếng tự động toàn bộ quy trình (S0 -> S9) bằng gTTS
python -m ecoup.cli dub -i video.mp4 --backend gtts --whisper-model large-v3

# Lồng tiếng với F5-TTS Voice Cloning (chất lượng cao trên GPU)
python -m ecoup.cli dub -i video.mp4 --backend f5 --whisper-model large-v3

# Cung cấp bối cảnh dịch thuật & xưng hô nhân vật
python -m ecoup.cli dub -i video.mp4 --context "Phim sitcom, nhân vật chính Peter Griffin hài hước mỉa mai"

# Chạy kiểm thử từng bước riêng lẻ (S1: Demucs, S2: Whisper, S3: Diarization, S5: Dịch, S8: Trộn)
python -m ecoup.cli dub -i video.mp4 --step s1
```

---

## 5. Tải trọng số mô hình (Model Weights)

```powershell
# Tải bộ trọng số SeedVR2-3B
python scripts/download_seedvr2.py

# Tải bộ trọng số CCSR v2 (ControlNet)
python scripts/download_ccsr.py
```

---

## 6. Đánh giá & Kiểm thử (Evaluation & Tests)

### Chạy toàn bộ bộ kiểm thử tự động
Toàn bộ 61 test cases kiểm tra video I/O, tiling, blending, dedup, checkpoint, clip chunking, pipeline upscale, trích xuất âm thanh, Whisper ASR, ngắt dòng phụ đề CPL/CPS, biên dịch SRT/ASS, bản dịch mock Gemini, và burn-in libass:
```powershell
pytest -v
```

### Điểm chuẩn tốc độ (Benchmarks)
Xem báo cáo chi tiết và số liệu thực nghiệm tại [`docs/benchmark.md`](file:///docs/benchmark.md):
- Phần 1 – 4: Đo kiểm VRAM, FPS và nhiệt độ cho các mô hình Upscale (SeedVR2, CCSR, NomosUni).
- Phần 5: Đo kiểm hiệu năng Pipeline Phụ đề đa ngôn ngữ (English, Chinese, Japanese, Korean, Vietnamese) trên Tesla T4 Colab (tốc độ tổng thể nhanh gấp 1.3x – 5.0x thời gian thực của video).

### Đánh giá so sánh đa mô hình (Side-by-Side Face Zoom & Full Frame)
```powershell
python scripts/eval_compare.py --models nomosuni seedvr2
```

### Thử nghiệm video dài (Long Video Stress Test 2.5 phút)
```powershell
python scripts/benchmark_long_video.py
```

---

## 7. Hạn chế đã biết & Lưu ý kỹ thuật

1. **Phiên bản thư viện CCSR**: Mã nguồn CCSR v2 yêu cầu chính xác `diffusers==0.21.4` và `transformers==4.35.2`. Phiên bản mới hơn sẽ gây lỗi chữ ký tham số trong `StableDiffusionControlNetPipeline`.
2. **Quản lý bộ nhớ VRAM cho SeedVR2**:
   - Ở chế độ `resident` trên GPU 24GB (L4/RTX 3090/4090), VRAM duy trì ở mức **16.19 GB** cho cửa sổ 5 khung (`sub_len: 5`).
   - Nếu chạy `model_resolution: match_target` trực tiếp ở 1440×2560 trên GPU 24GB sẽ gây lỗi tràn VRAM (CUDA OOM) tại VAE. Do đó, hệ thống mặc định chọn `model_resolution: native` (xử lý nội bộ 720p rồi nội suy Lanczos4 2x) và tự động bắt lỗi OOM để fallback an toàn.
   - Trên Tesla T4 16GB, bắt buộc dùng `memory_mode: offload` (VRAM đỉnh 9.85 GB) để tránh OOM.
3. **Quy tắc cửa sổ thời gian $4k+1$**: Tham số `sub_len` của SeedVR2 tuân theo quy tắc số khung $4k+1$ ($5, 9, 13, 17, \dots$) để bảo toàn tính đồng nhất thời gian của khối Causal 3D Conv và Rotary Position Embedding (RoPE).
4. **Bảo toàn số khung hình & Âm thanh**: Hệ thống đảm bảo 100% số lượng khung hình đầu ra khớp chính xác với đầu vào thông qua thuật toán cắt lớp gối đầu (overlap slicing) và hòa trộn tuyến tính (cross-fade), đồng thời tự động remux track âm thanh gốc (AAC/MP3) vào file thành phẩm.
5. **Khả năng lặp lại với Seed cố định**: Seed ngẫu nhiên mặc định là `42` (hoặc cấu hình tùy ý/ngẫu nhiên). Mỗi tác vụ xuất ra file siêu dữ liệu cấu hình tương ứng `output_video.json` đặt cạnh video đầu ra nhằm hỗ trợ lặp lại chính xác kết quả thí nghiệm.
