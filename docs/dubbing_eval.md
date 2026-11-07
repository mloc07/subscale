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
