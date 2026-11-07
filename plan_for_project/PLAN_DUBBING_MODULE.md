# KẾ HOẠCH: Module lồng tiếng AI nhiều nhân vật (tab "Lồng tiếng")

> Tài liệu dành cho agent (Antigravity) thực thi, có thể chạy thử trực tiếp trên Google Colab. Làm tuần tự theo Phase; hoàn thành **Tiêu chí nghiệm thu** của Phase trước khi sang Phase sau. Sau mỗi Phase: chạy `pytest`, rồi commit.
> **Mọi tên model, thư viện, giấy phép, yêu cầu VRAM trong tài liệu này chỉ là ứng viên chưa kiểm chứng.** Phase 0 bắt buộc xác minh trong tài liệu/repo gốc hiện hành; không dựa vào trí nhớ. Mọi con số hiệu năng phải đo thật; nếu chưa đo được thì ghi "chưa đo".

---

## 1. Mục tiêu

Biến một video có lời thoại (ví dụ tiếng Anh) thành bản **lồng tiếng Việt** tự động, giữ được:
- **Từng nhân vật có giọng riêng**, âm sắc bám theo giọng gốc của chính nhân vật đó.
- **Nhạc nền và hiệu ứng âm thanh gốc** được giữ nguyên (không bị mất tiếng nổ, bước chân, nhạc).
- **Thời lượng từng câu khớp** với câu gốc (không tràn sang câu sau, không ngắt quãng).
- Video hình ảnh **không bị mã hóa lại** khi chỉ thay âm thanh.

Quy trình chạy tự động từ đầu đến cuối bằng một lần bấm, nhưng **mọi bước đều lưu sản phẩm trung gian** để có thể xem lại, sửa và chạy tiếp từ giữa chừng.

## 2. Phạm vi

**Làm:** tách âm, nhận dạng, nhận dạng người nói, hồ sơ giọng từng nhân vật, dịch có ngân sách thời lượng, tạo giọng, khớp thời lượng, trộn âm, ghép video, báo cáo chất lượng, tab giao diện riêng, CLI.

**Không làm trong kế hoạch này:** đồng bộ khẩu hình (lip-sync), lồng tiếng thời gian thực, chuyển giọng cho nội dung trực tiếp, tạo giọng nhân vật có sẵn đóng gói trong repo.

## 3. Bối cảnh mã nguồn (cần đọc trước khi làm)

- Package `ecoup` ở `src/ecoup/`. Đã có module phụ đề `src/ecoup/subtitle/` gồm: `audio.py` (tách âm thanh, `extract_vocals` dùng Demucs), `asr.py` (faster-whisper), `segments.py`, `translate/` (`base.py`, `gemini.py`), `srt.py`, `burn.py`, `snapping.py`, `pipeline.py`, `report.py`, `config.py`.
- Giao diện Colab: `colab_app.py` (Gradio). GUI máy tính: `gui/main_window.py` (PyQt6). Cần thêm một tab "Lồng tiếng" ở cả hai.
- Các vấn đề đã quan sát ở module phụ đề, **liên quan trực tiếp đến lồng tiếng và nên xử lý trước hoặc cùng lúc** (xác minh lại bằng đọc mã):
  1. `extract_vocals()` nhận file WAV **16 kHz mono** đã tách sẵn, trong khi Demucs được huấn luyện cho 44,1 kHz stereo; nên đưa âm thanh gốc chất lượng cao vào Demucs.
  2. Khi Demucs lỗi, pipeline chỉ ghi cảnh báo rồi chạy tiếp với âm thanh chưa tách. Với lồng tiếng, **thất bại tách âm phải dừng rõ ràng**, vì kết quả nhạc nền sẽ sai.
  3. Chế độ `language=None` có thể khóa một ngôn ngữ cho cả file; `language_whitelist` có thể chỉ đổi nhãn chứ không giới hạn giải mã. Cần ngôn ngữ đúng theo từng đoạn.
- Dùng lại tối đa: ASR, `Segment`, dịch Gemini (thêm ngân sách thời lượng), kiểm tra FFmpeg, cơ chế báo cáo, checkpoint/cache.
- Đặt mã mới trong `src/ecoup/dubbing/`, độc lập tên (repo có thể đổi tên sau).

## 4. Ràng buộc và quy tắc (BẮT BUỘC)

**Về nhân bản giọng và bản quyền:**
- **Không** đóng gói sẵn giọng mẫu của người thật hoặc nhân vật có bản quyền trong repo.
- **Không chèn watermark hay metadata cưỡng ép**: Video đầu ra giữ nguyên vẹn chất lượng và thuộc tính theo yêu cầu người dùng, không gắn watermark hay siêu dữ liệu "lồng tiếng bằng AI". Không yêu cầu ô xác nhận quyền trên giao diện.
- Ghi rõ trong README mục "Sử dụng có trách nhiệm" (không mạo danh, không tạo nội dung đánh lừa).

**Về kỹ thuật:**
- Python 3.10+; giữ phong cách mã hiện có; không phá vỡ test và chức năng upscale/phụ đề.
- Tham số cấu hình được qua dataclass (kèm `validate()`); không hard-code.
- **Khóa API không vào repo**; đọc từ biến môi trường hoặc Colab Secrets; không ghi vào log.
- **VRAM:** Demucs, Whisper, mô hình nhận dạng người nói, TTS, chuyển giọng và SeedVR2 **không chạy cùng lúc**; nạp, dùng, giải phóng.
- **Xung đột thư viện:** các bộ TTS/chuyển giọng thường ghim phiên bản `torch`/`numpy` riêng. Mỗi backend nằm sau một interface, là **phụ thuộc tùy chọn (extras)**, và nếu xung đột thì chạy trong tiến trình/môi trường riêng (subprocess + file trung gian). Không bắt người chỉ dùng upscale phải cài.
- Unit test không gọi model hay API thật (dùng backend giả).
- Không commit video, âm thanh, trọng số model, khóa API hoặc file lớn.

## 5. Quy trình chuyên nghiệp (tổng quan)

```
Video gốc
  │
  ▼  S0  Tiếp nhận & kiểm tra (ffprobe, lưu manifest.json)
  │
  ▼  S1  Tách âm: lời thoại (dialogue.wav) | nhạc nền + hiệu ứng (bgm.wav)
  │
  ▼  S2  Nhận dạng giọng nói (ASR) trên dialogue.wav, ngôn ngữ đúng từng đoạn
  │
  ▼  S3  Nhận dạng người nói + phát hiện nói chồng  →  đoạn có nhãn nhân vật
  │
  ▼  S4  Hồ sơ giọng từng nhân vật (chọn đoạn tham chiếu sạch)
  │
  ▼  S5  Dịch theo ngữ cảnh, có ngân sách thời lượng, biết giới tính/xưng hô từng người
  │
  ▼  S6  Tạo giọng tiếng Việt theo từng nhân vật  (+ tùy chọn chuyển âm sắc)
  │
  ▼  S7  Khớp thời lượng (tốc độ / viết lại ngắn hơn / đệm im lặng)
  │
  ▼  S8  Ráp theo mốc thời gian + trộn với nhạc nền (ducking, chuẩn hóa âm lượng)
  │
  ▼  S9  Ghép vào video (sao chép luồng hình, không mã hóa lại)
  │
  ▼  S10 Báo cáo chất lượng + cờ cảnh báo
```

Mỗi bước đọc đầu vào từ và ghi đầu ra vào **thư mục dự án** (mục 6). Có thể chạy lại từ bất kỳ bước nào mà không làm lại các bước trước.

## 6. Dữ liệu và thư mục dự án

```
<project>/
  manifest.json        # trạng thái từng bước, cấu hình đã dùng, seed, phiên bản, thời gian
  audio/
    original_hq.wav    # 44,1 kHz stereo trích từ video gốc
    dialogue.wav       # lời thoại sau tách (S1)
    bgm.wav            # nhạc nền + hiệu ứng (S1)
  transcript/
    segments.json      # ASR: start, end, text, language, confidence
    diarized.json      # + speaker_id, overlap flag
  voices/
    <speaker_id>/ref_*.wav, profile.json   # đoạn tham chiếu + tên nhân vật + giới tính (nếu có)
  translation/
    translated.json    # text_vi, ngân sách thời lượng, số lần viết lại, cờ
  tts/
    <segment_id>.wav   # giọng tạo ra (và bản sau chuyển giọng nếu có)
  mix/
    dub_mix.wav
  out/
    <tên>_dub_vi.mp4, *.srt, report.json
```

`manifest.json` lưu băm cấu hình từng bước để biết bước nào cần chạy lại khi đổi tham số; cache theo băm.

## 7. Chi tiết từng bước

**S0 – Tiếp nhận.** Kiểm tra video có âm thanh; ghi độ dài, fps, kênh âm thanh; trích `original_hq.wav` (44,1 kHz stereo) bằng FFmpeg. Từ chối rõ ràng nếu không có âm thanh.

**S1 – Tách âm.** Dùng Demucs (hoặc công cụ tương đương nếu Phase 0 tìm được tốt hơn) trên `original_hq.wav`. Đầu ra: `dialogue.wav`, `bgm.wav`. Nếu thất bại thì **dừng**, không chạy tiếp. Giải phóng VRAM sau bước này.

**S2 – ASR.** Whisper trên `dialogue.wav`. Ngôn ngữ **theo từng đoạn/cửa sổ** (không khóa một ngôn ngữ cho cả file); nếu người dùng cung cấp danh sách ngôn ngữ cho phép thì giới hạn ngay lúc giải mã/chọn ngôn ngữ, không chỉ đổi nhãn. Lọc đoạn nghi bịa (không giao với vùng tiếng nói theo VAD, cụm lặp, cụm rác quen thuộc). Lưu `segments.json` có độ tin cậy và mốc thời gian theo từ khi có.

**S3 – Nhận dạng người nói.**
- Dùng mô hình nhận dạng người nói (diarization) + nhúng giọng nói (speaker embedding) rồi phân cụm; **không** dùng riêng tần số/cao độ làm tiêu chí.
- Tùy chọn người dùng: số người nói, và/hoặc **mẫu giọng đăng ký (enrollment)** cho từng nhân vật để neo nhãn.
- **Phát hiện nói chồng** và gắn cờ `overlap`; đoạn chồng xử lý riêng (chọn người nói chính, hoặc đánh dấu để người dùng xem), không "tắt âm phần còn lại" một cách mù quáng.
- Cho phép **gộp/tách cụm** và **đặt tên nhân vật** qua bảng (mặc định tắt bước xem lại; tự động hoàn toàn vẫn chạy được).
- Cảnh báo khi các nhân vật khác nhau có thể cùng một người lồng tiếng (cụm bị gộp) hoặc một người bị tách thành nhiều cụm.
- Biểu diễn kết quả là **danh sách đoạn theo từng nhân vật và một track theo mốc thời gian**, không sinh video con riêng cho mỗi nhân vật. Cung cấp chế độ **xem trước/nghe riêng từng nhân vật** (xuất âm thanh hoặc video ngắn) khi cần.

**S4 – Hồ sơ giọng.** Cho mỗi nhân vật, chọn vài đoạn tham chiếu sạch dựa trên: thời lượng đủ dài, không nói chồng, ít nhiễu/nhạc dư, năng lượng ổn định, không có tiếng cười/hét. Ghép thành mẫu tham chiếu theo yêu cầu của backend (độ dài tối thiểu/tối đa cấu hình được). Lưu điểm chất lượng; cảnh báo nếu nhân vật không có đủ giọng sạch (khi đó cho phép đổi sang giọng có sẵn do người dùng chọn).

**S5 – Dịch.** Tái dùng `translate/` (Gemini, cấu hình model, không hard-code). Mở rộng:
- Biết **người nói** của từng câu (đại từ xưng hô, giới tính, quan hệ) qua hồ sơ nhân vật và ghi chú ngữ cảnh/bảng thuật ngữ do người dùng nhập.
- **Ngân sách thời lượng**: với mỗi câu, tính số ký tự/âm tiết tối đa từ thời lượng gốc và tốc độ nói tiếng Việt mục tiêu; yêu cầu bản dịch **thoát ý nhưng ngắn gọn** trong ngân sách, giữ giọng điệu và hài hước.
- Dịch theo lô có ngữ cảnh gối đầu, đầu ra có cấu trúc (JSON), kiểm tra số đoạn khớp; cache theo lô.
- Quy tắc nội dung nhạy cảm: prompt dịch thuật bình thường; **không** dùng mẹo đánh lừa để vượt bộ lọc của model; xử lý bị chặn bằng thử lại lô nhỏ hơn, backend dự phòng, hoặc giữ bản gốc kèm cờ cảnh báo (như module phụ đề).

**S6 – Tạo giọng.**
- Interface `VoiceBackend`: nhận (văn bản tiếng Việt, hồ sơ giọng nhân vật, ngôn ngữ, tham số tốc độ), trả về file WAV. Tách riêng interface `TimbreConverter` (tùy chọn): nhận (WAV, hồ sơ giọng) → WAV có âm sắc của nhân vật.
- Hai hướng cần so sánh ở Phase 0/7 (không chọn trước): (a) TTS zero-shot theo giọng mẫu trực tiếp; (b) TTS tạo ngữ điệu tiếng Việt rồi chuyển âm sắc sang giọng gốc.
- **Không** dùng dịch cao độ (pitch shift) đơn thuần làm cách chính để "đổi tone cho giống gốc" vì âm sắc không đổi theo; chỉ dùng như hiệu chỉnh phụ nếu số đo cho thấy có ích.
- Chuẩn hóa văn bản trước TTS (số, viết tắt, tên riêng, dấu tiếng Việt).

**S7 – Khớp thời lượng.** Với mỗi câu: đo thời lượng giọng tạo ra so với khoảng cho phép.
1. Trong dung sai → giữ.
2. Quá dài → tăng tốc trong giới hạn cấu hình (ví dụ khoảng an toàn do Phase 5 đo, không đặt tùy ý), nếu vẫn quá → gọi lại S5 với ngân sách chặt hơn (tối đa N lần).
3. Vẫn quá → cho phép tràn vào khoảng lặng liền sau (nếu không có người khác nói), nếu không → giữ bản nhanh nhất và **gắn cờ**.
4. Quá ngắn → đệm im lặng theo mốc bắt đầu gốc, không kéo giãn giọng bất thường.
Giữ **mốc bắt đầu** theo gốc làm chuẩn.

**S8 – Ráp và trộn.** Đặt từng đoạn theo mốc thời gian vào track lời thoại mới; cân bằng độ to từng nhân vật theo độ to của giọng gốc; **giảm nhạc nền khi có thoại (ducking)** nhẹ, giữ hiệu ứng; chuẩn hóa tổng âm lượng theo mục tiêu cấu hình; làm mờ nối (crossfade) ở chỗ ghép; tránh méo/clipping. Xuất `dub_mix.wav`.

**S9 – Ghép video.** Thay âm thanh bằng FFmpeg **sao chép luồng hình (`-c:v copy`)**, không mã hóa lại hình; tùy chọn giữ âm thanh gốc làm track phụ; xuất `.srt` tiếng Việt (tái dùng module phụ đề) kèm tùy chọn (không thêm watermark hay metadata cưỡng ép).

**S10 – Báo cáo.** `report.json` và bản tóm tắt: số nhân vật, số đoạn, số đoạn bị cờ (thời lượng, nói chồng, giọng mẫu kém, dịch bị chặn), thời gian từng bước, VRAM đỉnh, tham số và backend đã dùng, cảnh báo cần người xem lại.

## 8. Phase thực hiện

### Phase 0: Khảo sát ứng viên và môi trường (bắt buộc làm trước)
1. Với mỗi nhóm, lập danh sách ứng viên và **xác minh trong tài liệu/repo gốc**: giấy phép (cho dùng cá nhân/thương mại?), yêu cầu VRAM, hỗ trợ **tiếng Việt**, cách cài trên Colab, xung đột phụ thuộc, có cần token/đồng ý điều khoản không:
   - Nhận dạng người nói + nhúng giọng nói.
   - TTS zero-shot hỗ trợ tiếng Việt (gồm các bản tinh chỉnh tiếng Việt do cộng đồng nếu có; **chỉ dùng khi xác minh được nguồn và giấy phép**) và dịch vụ trả phí qua API nếu có.
   - Chuyển âm sắc (voice conversion) zero-shot hoặc huấn luyện theo giọng.
   - Công cụ tách âm (kiểm tra bản Demucs đang dùng và lựa chọn thay thế).
2. Ghi kết quả vào `docs/dubbing_notes.md` (ngày kiểm tra, nguồn).
3. Dựng **bộ đo** (`scripts/dub_eval.py`): độ giống giọng (cosine giữa nhúng giọng gốc và giọng tạo), độ rõ chữ (đọc ngược giọng tạo bằng ASR rồi tính tỉ lệ lỗi so với văn bản dịch), sai lệch thời lượng từng câu, và tỉ lệ lỗi phân người nói trên một clip có nhãn tay (DER hoặc số đoạn gán sai).
4. Chuẩn bị 2–3 clip thử (một người nói; hai người nói luân phiên; có nhạc nền và nói chồng nhau). **Hỏi người dùng** cung cấp clip và xác nhận họ có quyền sử dụng; không dùng nội dung người dùng không có quyền.

**Nghiệm thu:** `docs/dubbing_notes.md` có bảng ứng viên đã xác minh; bộ đo chạy được trên backend giả; có clip thử được người dùng xác nhận.

### Phase 1: Nền tảng âm thanh và manifest
1. Tạo cấu trúc thư mục dự án, `manifest.json`, cơ chế chạy lại từ bước bất kỳ và cache theo băm cấu hình.
2. S0 và S1: trích `original_hq.wav` (44,1 kHz stereo), tách âm (`dialogue.wav`, `bgm.wav`); thất bại thì dừng rõ ràng.
3. Sửa `extract_vocals` của module phụ đề để dùng âm thanh chất lượng cao (nếu chưa), giữ tương thích ngược.
4. Test dùng engine tách âm giả.

**Nghiệm thu:** chạy S0–S1 trên clip thử cho ra hai track nghe được; ngắt và chạy lại không làm lại bước đã xong; `pytest` xanh.

### Phase 2: ASR đúng ngôn ngữ và nhận dạng người nói
1. S2: ASR theo ngôn ngữ từng đoạn, lọc đoạn nghi bịa, ghi `segments.json`.
2. S3: phân cụm người nói, enrollment tùy chọn, phát hiện nói chồng, gộp/tách cụm và đặt tên (qua cấu hình/CLI trước, UI sau).
3. Xuất chế độ nghe riêng từng nhân vật (âm thanh).
4. Đo tỉ lệ lỗi phân người nói trên clip có nhãn tay, ghi vào `docs/dubbing_eval.md`.

**Nghiệm thu:** có `diarized.json` đúng định dạng; có số đo lỗi phân người nói; người dùng nghe thử và xác nhận chất lượng phân tách đủ dùng (hoặc ghi rõ giới hạn).

### Phase 3: Hồ sơ giọng
1. S4: chọn đoạn tham chiếu sạch, tính điểm chất lượng, xuất `voices/<id>/`.
2. Cảnh báo khi giọng mẫu kém; cho phép người dùng cung cấp mẫu giọng riêng.
3. Test với dữ liệu giả.

**Nghiệm thu:** mỗi nhân vật trong clip thử có hồ sơ giọng với điểm chất lượng; đoạn kém được gắn cờ.

### Phase 4: Dịch có ngân sách thời lượng
1. Mở rộng S5 theo mục 7; thêm vòng viết lại ngắn hơn có giới hạn số lần.
2. Test bằng backend dịch giả: số đoạn khớp, ngân sách được tôn trọng, cache hoạt động, bị chặn xử lý đúng, khóa API không xuất hiện trong log.

**Nghiệm thu:** bản dịch của clip thử nằm trong ngân sách thời lượng ở đa số câu; số câu phải viết lại và số câu bị cờ được báo cáo.

### Phase 5: MVP lồng tiếng một người nói (cột mốc quan trọng)
1. Cài backend TTS đã chọn ở Phase 0 sau interface `VoiceBackend`.
2. S6 (hướng a), S7, S8, S9 cho clip **một người nói**.
3. Đo: độ giống giọng, độ rõ chữ, sai lệch thời lượng; ghi dải tốc độ an toàn thực tế vào cấu hình mặc định.
4. Người dùng nghe và đánh giá. **Dừng và báo cáo** trước khi sang Phase 6.

**Nghiệm thu:** video lồng tiếng chạy được từ đầu đến cuối, nhạc nền còn nguyên, hình không bị mã hóa lại, đồng bộ chấp nhận được; có số đo và đánh giá của người dùng.

### Phase 6: Nhiều nhân vật
1. Mỗi nhân vật dùng hồ sơ giọng riêng; xử lý đoạn nói chồng; cân bằng độ to giữa các nhân vật.
2. Cảnh báo tình huống gộp/tách nhầm cụm.
3. Chạy clip thử hai người nói luân phiên và clip có nói chồng; ghi kết quả.

**Nghiệm thu:** clip nhiều nhân vật cho ra giọng khác nhau đúng nhân vật; các đoạn bị cờ được liệt kê trong báo cáo.

### Phase 7: Chuyển âm sắc và bám giọng điệu (thí nghiệm)
1. Cài `TimbreConverter` sau interface; so sánh hướng (a) và (b) bằng độ giống giọng, độ rõ chữ, và nghe thử mù.
2. Thử hiệu chỉnh cao độ nhẹ chỉ như bước phụ, chỉ giữ nếu số đo cho thấy có lợi.
3. Chọn mặc định theo số đo; ghi rõ trường hợp hướng nào tốt hơn.

**Nghiệm thu:** bảng so sánh có số đo, và đề xuất mặc định có lý do. Nếu chưa vượt hướng (a), giữ (a) và ghi rõ.

### Phase 8: Giao diện và CLI
1. Tab "Lồng tiếng" riêng biệt trên Colab (nằm cạnh tab "Upscale" và "Phụ đề" trong `gr.Tabs`) và PyQt (`QTabWidget`): chọn video, ngôn ngữ nguồn, số/tên nhân vật (tùy chọn), mẫu giọng đăng ký (tùy chọn), backend giọng, model dịch, ghi chú ngữ cảnh/bảng thuật ngữ, nút chạy/hủy, tiến độ từng bước, nghe thử từng nhân vật, tải kết quả và báo cáo. Không có watermark hay nút xác nhận quyền rườm rà.
2. CLI lệnh con `dub` hỗ trợ chạy từ bước bất kỳ.
3. Chạy ở luồng riêng, hủy an toàn, giải phóng GPU sau mỗi bước.
4. Bước "xem lại và sửa" (bảng phân người nói/bản dịch) là tùy chọn, mặc định tắt.

**Nghiệm thu:** chạy trọn một lượt trên Colab từ giao diện tab "Lồng tiếng" mà không cần can thiệp tay; hủy giữa chừng không để lại tiến trình treo.

### Phase 9: Kiểm thử, tài liệu, video dài
1. Test đơn vị đầy đủ bằng backend giả (khớp thời lượng, ráp mốc, trộn, ghi manifest, chạy lại từ bước bất kỳ, tính cờ).
2. Chạy video dài vài phút trên Colab: đo thời gian từng bước, VRAM đỉnh, hành vi khi ngắt và tiếp tục; cập nhật `docs/benchmark.md`.
3. README: cài đặt (extras), cách đặt khóa API, mục "Sử dụng có trách nhiệm", giới hạn đã biết (nói chồng, nhân vật hoạt hình cường điệu, cảm xúc, tiếng Việt).

**Nghiệm thu:** README và báo cáo đầy đủ; người dùng xác nhận kết quả.

### Phase 10 (để sau, không làm trong đợt này)
Đồng bộ khẩu hình; tái hiện cảm xúc nâng cao; tự động hiệu chỉnh theo từng cảnh.

## 9. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Phân nhầm/gộp nhầm nhân vật (đặc biệt cùng một người lồng tiếng) | Nhúng giọng + enrollment + bảng gộp/tách + cảnh báo |
| Nói chồng, tiếng cười, tiếng ồn làm giọng mẫu kém | Chọn đoạn sạch, điểm chất lượng, cờ cảnh báo, cho phép mẫu do người dùng cung cấp |
| Câu tiếng Việt không vừa thời lượng | Ngân sách trong dịch, tăng tốc có giới hạn, viết lại, đệm lặng, cờ |
| Giọng tạo ra không giống/không đủ cảm xúc | So hai hướng bằng số đo; đặt kỳ vọng thực tế; xem lại thủ công |
| Model TTS/VC không hỗ trợ tiếng Việt tốt hoặc giấy phép không cho phép | Phase 0 xác minh; có backend dự phòng; ghi giới hạn |
| Xung đột thư viện trên Colab | Backend tùy chọn, tiến trình riêng, file trung gian |
| Hết VRAM | Chạy tuần tự, giải phóng giữa bước, theo dõi VRAM đỉnh |
| Colab ngắt phiên | Manifest + cache + chạy lại từ bước bất kỳ |
| Rủi ro bản quyền giọng nói | Không đóng gói sẵn giọng người thật trong mã nguồn, tài liệu hướng dẫn rõ ràng |
| Nội dung nhạy cảm bị API dịch chặn | Xử lý có kiểm soát, giữ bản gốc kèm cờ; không dùng mẹo vượt bộ lọc |

## 10. Quy tắc làm việc cho agent

- Làm đúng thứ tự Phase; **Phase 0 trước**. **Dừng và hỏi người dùng** ở cuối Phase 0 (chọn ứng viên), cuối Phase 5 (đánh giá MVP) và khi cần quyết định lớn ảnh hưởng chất lượng hoặc thêm phụ thuộc nặng.
- Mọi "giả thuyết" và tên model trong tài liệu này là chưa kiểm chứng: xác minh bằng tài liệu gốc hoặc thí nghiệm, tin số đo hơn giả thuyết và báo cáo khi khác.
- Không viết prompt hay mã nhằm vượt bộ lọc an toàn của model; tôn trọng cấu hình người dùng (không chèn watermark hay popup xác nhận gây cản trở trải nghiệm).
- Không thêm tính năng ngoài phạm vi (lip-sync, thời gian thực).
- Báo cáo ngắn sau mỗi Phase: đã làm gì, số liệu đo, vấn đề còn lại.
- Không commit video, âm thanh, trọng số model, khóa API hoặc file lớn.
