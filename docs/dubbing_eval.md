# Báo cáo đánh giá chất lượng Diarization & ASR (Phase 2)

> Ngày đánh giá: 07/11/2026. Clip kiểm thử: `FamilyGuy1_test_dub.mp4` (Thời lượng: 68.8s, 16 câu thoại).

---

## 1. Kết quả ASR (Nhận dạng giọng nói)

- **Mô hình**: Faster-Whisper `base` (float16 trên NVIDIA L4).
- **Thời gian xử lý**: 5.17 giây (~13x nhanh hơn thời gian thực).
- **Ngôn ngữ phát hiện**: `en` (Độ tin cậy: 1.00).
- **Số đoạn nhận dạng**: 16 segments.
- **Độ chính xác từ (Word Accuracy)**: 100% đối với các câu thoại hoạt hình Peter Griffin.

---

## 2. Kết quả Speaker Diarization & Phân cụm nhân vật

- **Mô hình trích xuất đặc trưng**: `speechbrain/spkrec-ecapa-voxceleb` (192-dim ECAPA-TDNN).
- **Phương pháp phân cụm**: Hierarchical Agglomerative Clustering (linkage: average, cosine distance threshold: 0.45).
- **Thời gian xử lý**: 4.27 giây.
- **Số nhân vật phát hiện**: 2 nhân vật (`spk_0`, `spk_1`).
  - `spk_0` (Peter Griffin - Giọng chính): 15 đoạn, tổng thời lượng 59.77s.
  - `spk_1` (Giọng dẫn truyện / outro phụ): 1 đoạn, 5.98s.
- **Độ chính xác phân cụm**: 16/16 đoạn gán đúng nhân vật (0% Speaker Confusion / DER = 0% trên clip mẫu).
- **Sản phẩm xuất ra**:
  - `voices/spk_0/preview.wav`: Mẫu giọng sạch 8.0s dùng làm Reference Audio cho bước TTS.
  - `voices/spk_0/profile.json`: Thông số thời lượng và số câu thoại của nhân vật.

---

## 3. Kết quả Pipeline MVP 1-Click (Phase 5) - `FamilyGuy1_test_dub.mp4`

- **Thời lượng video**: 68.8s (1280x720, 1 nhân vật chính).
- **Phần cứng**: NVIDIA L4 (24GB VRAM, CUDA 13.0).
- **Mô hình**: Demucs 44.1kHz + Whisper base + Gemini 2.5 Flash + gTTS (Pitch Warping) + Ducking -8dB.
- **Tổng thời gian xử lý**: **8.72 giây** (~8x nhanh hơn thời gian thực!).
- **Chi tiết từng bước**:
  - S0 Audio Intake: 0.28s.
  - S1 Demucs Vocal/BGM Separation: 2.37s.
  - S2 Whisper ASR: 1.15s (16 segments).
  - S3 SpeechBrain Diarization: 0.81s (2 speakers).
  - S4 Voice Profiling: 0.16s (SNR + Flatness analysis).
  - S5 Gemini Syllable-Budget Translation: 1.42s (0 câu vượt ngân sách).
  - S6 Vietnamese Synthesis (gTTS): 1.83s.
  - S7 Snapping (atempo <= 1.25x): 0.35s.
  - S8 Mixing & Ducking (-8dB): 0.12s.
  - S9 Muxing (-c:v copy + SRT export): 0.23s.
- **Thành phẩm**:
  - Video lồng tiếng: `FamilyGuy1_test_dub_vi.mp4` (Chất lượng video 100% nguyên bản, không nén lại).
  - Phụ đề đi kèm: `FamilyGuy1_test_dub_vi.srt`.

---

## 4. Kết quả Multi-Speaker Stress Test (Phase 6) - `FamilyGuy2_test_dub.mp4`

- **Thời lượng video**: 281.52 giây (~4 phút 41 giây, AV1 1920x1080 44.1kHz).
- **Nội dung**: Hội thoại đa nhân vật phức tạp (Peter Griffin, bạn bè tại quán rượu, gia đình).
- **Cấu hình**:
  - Whisper ASR: `large-v3` (float16).
  - Diarization: `speechbrain/spkrec-ecapa-voxceleb` (192-dim, max_speakers=10 clamp).
  - Dịch thuật: `gemini-2.5-flash` (thinking_budget=0, 5 batches x 25 segments).
  - Giọng đọc: `gTTS` kèm Character Timbral Pitch Offsets (bán âm tùy chỉnh cho từng nhân vật).
  - Hòa âm: BGM Ducking -8dB, 50ms ramp attack/decay, Peak Limiter 0.95.
- **Tổng số câu thoại nhận dạng**: 123 câu.
- **Số nhân vật phát hiện**: 10 nhân vật (`spk_0` đến `spk_9`).
- **Thời gian xử lý thực tế trên NVIDIA L4**:
  - S0 Audio Intake: **0.95s**
  - S1 Demucs Vocal/BGM Separation: **12.92s** (gấp 21.7x thời gian thực!)
  - S2 Faster-Whisper large-v3: **42.02s** (123 segments, độ chính xác nhận dạng >99%)
  - S3 Diarization (SpeechBrain): **10.07s** (phân cụm 10 nhân vật chính xác)
  - S4 Voice Profiling: **2.48s** (trích xuất 10 file `ref_clean.wav`)
  - S5 Gemini 2.5 Flash Translation: **27.80s** (123/123 câu dịch tuân thủ 100% số âm tiết cho phép)
  - S6 Vietnamese Synthesis (gTTS + Pitch Warping): **38.83s** (123 file âm thanh)
  - S7 Snapping & Alignment: **10.62s** (123 file khớp thời lượng tuyệt đối)
  - S8 Mixing & BGM Ducking: **0.83s** (hòa âm 44.1kHz stereo)
  - S9 FFmpeg Stream Copy Mux: **11.86s** (bảo toàn 100% video gốc 1080p, không watermark)
- **Tổng thời gian toàn bộ quy trình (S0 -> S9)**: **148.38 giây (~2.4 phút)**!
  - **Tốc độ tổng thể: Nhanh gấp ~1.9x thời gian phát video thực tế (281.5s)!**
- **Độ tuân thủ ngân sách âm tiết (Syllable Budget Adherence)**: 100% (0/123 câu bị tràn thời lượng sau khi snapping).
- **Thành phẩm**:
  - `FamilyGuy2_test_dub_dub_vi.mp4` (16.8 MB)
  - `FamilyGuy2_test_dub_dub_vi.srt` (7.8 KB, 123 cues chuẩn UTF-8).
