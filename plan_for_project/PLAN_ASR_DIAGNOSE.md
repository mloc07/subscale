# HƯỚNG DẪN CHẨN ĐOÁN VÀ SỬA TẠM THỜI: Phụ đề Whisper còn sai

> Dành cho agent (Antigravity) có thể **chạy test trực tiếp trên Google Colab**. Mục tiêu: tìm nguyên nhân thật bằng thí nghiệm có đo đạc, rồi áp dụng các sửa tạm an toàn. Các "nghi vấn" dưới đây là **suy luận từ đọc mã, chưa được kiểm chứng**: hãy xác minh bằng thí nghiệm, đừng coi là sự thật. Nếu số đo mâu thuẫn với giả thuyết, tin số đo và báo lại.
> Phạm vi: chỉ phần ASR (`src/ecoup/subtitle/asr.py`, `audio.py`, `segments.py`, `config.py`, `pipeline.py`). Không đổi phần upscale/SeedVR2.

---

## 0. Việc cần người dùng cung cấp trước (hỏi nếu chưa có)

1. **Một clip 1–2 phút bị sai phụ đề**, kèm ngôn ngữ và loại nội dung (hội thoại, vlog, có nhạc nền lớn, hát).
2. **Lời thoại đúng (reference)** do người dùng gõ tay cho clip đó (dạng văn bản, không cần mốc thời gian). Nếu chưa có thì dùng chế độ "so sánh các biến thể bằng mắt" (mục 3.4) và báo rõ là chưa có số đo WER/CER thật.
3. **Mô tả lỗi**: nghe nhầm từ / bịa chữ ở chỗ im lặng hoặc nhạc / lặp câu / lệch thời gian / dịch sai dù phiên âm đúng.

Không được tự bịa reference. Nếu dùng một model khác (ví dụ Gemini nghe âm thanh) làm "tham chiếu giả", phải gắn nhãn rõ là **không đáng tin tuyệt đối**.

## 1. Các giả thuyết (xếp theo mức nghi ngờ)

| # | Giả thuyết | Chỗ trong mã | Dấu hiệu nếu đúng |
|---|---|---|---|
| H1 | Âm thanh lẫn nhạc/tiếng ồn, Whisper nghe sai | `audio.py::extract_audio` chỉ tách WAV mono 16kHz | Lỗi tăng ở đoạn có nhạc/ồn; tách giọng xong thì giảm rõ |
| H2 | VAD quá nhạy gây bịa chữ | `audio_mode="music"` mặc định, `vad_threshold` 0.25 | Có chữ ở đoạn im lặng/nhạc; đổi sang `speech` thì giảm |
| H3 | Chia đoạn cố định 25s cắt giữa câu/từ | `asr.py` nhánh `multilingual_chunking` | Lỗi/mất/lặp chữ tập trung quanh mốc 25s, 50s, 75s... |
| H4 | Nhận diện ngôn ngữ từng đoạn sai/ép một ngôn ngữ cho cả video | `asr.py` (`detected_langs`), `pipeline.py` gán `s.language = res.language` | Video trộn ngôn ngữ bị dịch sai; ngôn ngữ đoạn khác nhau trong log |
| H5 | Mốc thời gian lệch do chia đoạn theo số ký tự | `segments.py::split_segment` | Phụ đề lệch nhịp giọng nói ở đoạn dài bị tách |
| H6 | Thiếu ngữ cảnh/từ vựng cho Whisper | `condition_on_previous_text=False`, không có `initial_prompt` | Tên riêng, thuật ngữ bị nghe sai lặp lại |
| H7 | Model/tham số chưa phù hợp | `large-v3`, `beam_size=5`, nhiệt độ mặc định | `large-v2`/`turbo` cho kết quả khác rõ rệt |
| H8 | Dịch sai chứ không phải phiên âm sai | `translate/gemini.py` | `.srt` gốc đúng nhưng `.vi.srt` sai |

**Bước đầu tiên bắt buộc:** xác định lỗi nằm ở **phiên âm (ASR)** hay ở **bản dịch (H8)** bằng cách so `.srt` gốc (`<tên>.<ngôn ngữ>.srt`) với reference. Nếu `.srt` gốc đúng, dừng chẩn đoán ASR và chuyển sang xem prompt/ngữ cảnh dịch.

## 2. Công cụ cần dựng trước khi thí nghiệm

1. `scripts/asr_eval.py`:
   - Nhận: đường dẫn video, file reference văn bản, danh sách cấu hình thí nghiệm.
   - Chạy từng cấu hình qua `FasterWhisperBackend`, lấy văn bản ASR, tính **WER** (hoặc **CER** cho tiếng Nhật/Trung/Hàn, vì không tách từ bằng khoảng trắng) sau khi chuẩn hóa (bỏ dấu câu, hạ chữ thường, chuẩn hóa Unicode NFKC).
   - Xuất bảng: tên cấu hình, WER/CER, số đoạn, thời gian chạy, VRAM đỉnh, ngôn ngữ phát hiện từng đoạn. Lưu vào `eval/asr/` (không commit video/âm thanh).
   - Dùng thư viện đo lỗi có sẵn (ví dụ `jiwer`) hoặc tự cài; thêm vào extras `dev`, không vào phụ thuộc chính.
2. Mỗi thí nghiệm lưu lại: cấu hình đầy đủ, `.srt` gốc, văn bản thô, log ngôn ngữ từng đoạn.
3. **Chỉ đổi MỘT biến mỗi lần** so với cấu hình nền (baseline = cấu hình mặc định hiện tại).

## 3. Thí nghiệm (theo thứ tự)

### 3.1 Phân loại lỗi (làm đầu tiên)
- Chạy baseline. Mở `.srt` gốc cạnh reference. Phân loại thủ công 30–50 lỗi đầu: **nghe nhầm**, **bịa chữ ở im lặng/nhạc**, **lặp**, **mất chữ**, **lệch thời gian**. Ghi tỉ lệ từng loại. Kết quả này định hướng thứ tự các bước sau.

### 3.2 Thí nghiệm theo giả thuyết
| Thí nghiệm | Biến đổi | Giả thuyết kiểm tra |
|---|---|---|
| E1 | `audio_mode`: `music` → `speech` → `off` | H2 |
| E2 | `multilingual_chunking`: bật → tắt (để Whisper xử lý cả file) | H3, H4 |
| E3 | Tách giọng khỏi nhạc (model chuyên dụng, ví dụ họ Demucs; xác minh gói và cách cài trên Colab) rồi đưa phần giọng vào Whisper | H1 |
| E4 | Thêm chuẩn hóa âm lượng + khử ồn nhẹ (FFmpeg `loudnorm`, `afftdn` hoặc tương đương) | H1 |
| E5 | `condition_on_previous_text`: False → True; thêm `initial_prompt` chứa tên riêng/thuật ngữ do người dùng cung cấp | H6 |
| E6 | Bật `word_timestamps` + `hallucination_silence_threshold` (kiểm tra tham số tồn tại trong phiên bản `faster-whisper` đang cài) | H2, H5 |
| E7 | Model: `large-v3` vs `large-v3-turbo` vs `large-v2` (tên kho model chính xác phải xác minh) | H7 |
| E8 | `beam_size` 1 / 5 / 8; `temperature` fallback mặc định vs tắt | H7 |

Ghi chú:
- E3 thường tốn thời gian và VRAM: chạy **tuần tự**, giải phóng model tách giọng trước khi nạp Whisper; đo VRAM đỉnh.
- Sau mỗi thí nghiệm đơn, thử **kết hợp các thay đổi đã có lợi** (E-combo) và xác nhận cải thiện cộng dồn, không giả định.
- Với tiếng Nhật/Hàn/Trung: tra cứu xem có model Whisper tinh chỉnh riêng cho ngôn ngữ đó mà tương thích `faster-whisper`/CTranslate2 không; chỉ thử nếu xác minh được nguồn, giấy phép và cách nạp. Báo cáo nếu không tìm được.

### 3.3 Kiểm tra riêng H3/H4 (chia đoạn và ngôn ngữ)
- Ghi vào log, cho từng đoạn 25s: ngôn ngữ phát hiện, xác suất, số đoạn trả về. Kiểm tra có đoạn nào đổi ngôn ngữ bất thường (nhận nhầm trên đoạn ồn/ngắn) không.
- Đối chiếu văn bản quanh mốc 25s/50s/... với reference để xem có mất/lặp chữ không.
- Kiểm tra `pipeline.py` có ghi đè ngôn ngữ từng đoạn bằng ngôn ngữ chung không, và translator có nhận đúng ngôn ngữ nguồn từng đoạn không.

### 3.4 Khi chưa có reference
- Chạy E1–E8 và cho người dùng nghe/xem **so sánh cạnh nhau** các `.srt` (bảng hiển thị từng dòng của các biến thể); người dùng chọn. Ghi rõ không có số WER/CER.

### 3.5 Ca thực tế đã có: bài hát/MV đa ngôn ngữ (Anh + Pidgin + Việt), có lời gốc

Bằng chứng người dùng đã cung cấp (so lời gốc với `.srt` do máy sinh): "Thanks for watching!" ở 00:00–00:28; "Baby, come near me now" (~00:28–00:50) bị mất; "Make me high like I'm on Tequila" bị nghe thành chuỗi vô nghĩa/thô tục; một cue tiếng Hàn `나 드셔` ngắn ở 02:05; đoạn Việt (verse ~02:30 và outro ~03:45) bị thay bằng câu kêu gọi subscribe kênh; cue bị kéo giãn 5–13 giây.

**Quan sát đã kiểm tra được từ chính file `.srt` (agent nên xác minh lại bằng log):**
- Nhiều cue bắt đầu đúng bội số 25 giây: 00:50, 01:15, 01:40, 02:05, 02:30, 03:20, 03:45 (= 50, 75, 100, 125, 150, 200, 225 s). Điều này khớp với chia cố định 25 giây của `multilingual_chunking`. Lỗi tập trung ở **đầu các chunk**; xem H3 là nghi vấn mạnh.
- Có những chunk **không trả gì** (ví dụ khoảng 00:28–00:50) dù có lời hát: mất nội dung thật, không chỉ sai chữ.
- Các đoạn tiếng Việt (có giọng hát thật) bị thay bằng câu "bịa", không phải im lặng. Vậy đây không chỉ là "bịa ở chỗ im lặng".

**Cảnh báo về đề xuất chỉnh ngưỡng (đã thấy trong một phản hồi trước đó, cần kiểm tra lại):**
- Trong cơ chế của Whisper, một đoạn chỉ bị bỏ khi `no_speech_prob > no_speech_threshold` **và** `avg_logprob < log_prob_threshold`. Vậy **tăng** `no_speech_threshold` (0.6 → 0.8) làm Whisper **ít** bỏ đoạn hơn, không phải "im lặng dứt khoát hơn". Muốn bỏ nhiều hơn thì thử **giảm** (ví dụ 0.4–0.5). Xác minh bằng thí nghiệm và đọc mã/tài liệu của phiên bản `faster-whisper` đang cài.
- `log_prob_threshold=-1.0` và `compression_ratio_threshold=2.4` thường **đã là giá trị mặc định**; đặt lại không đổi gì. Kiểm tra mặc định trong phiên bản đang cài.

**Thí nghiệm bổ sung cho nội dung bài hát (làm sau E1–E8, vẫn mỗi lần đổi một biến):**
| Thí nghiệm | Biến đổi |
|---|---|
| S1 | Tách giọng hát (Demucs hoặc tương đương, xác minh gói/cách cài trên Colab) rồi đưa `vocals.wav` vào Whisper. Giữ mốc thời gian theo video gốc. |
| S2 | Tắt chia cố định 25s; để Whisper xử lý cả file (hoặc chia theo khoảng lặng có gối đầu). So sánh số đoạn mất/bịa quanh bội số 25s. |
| S3 | Giới hạn tập ngôn ngữ được phép khi tự nhận diện (ví dụ chỉ `en`, `vi`), bằng cách chọn ngôn ngữ có xác suất cao nhất **trong tập cho phép**. Ghi log ngôn ngữ từng chunk để xem có chunk nào nhận `ko`/`zh` bất thường không. |
| S4 | Lọc đoạn nghi bịa: bỏ đoạn gần như không giao với vùng tiếng nói theo VAD, bỏ cụm "rác" quen thuộc (danh sách nhỏ, cấu hình được) và cụm lặp y hệt ở nhiều đầu chunk. **Chỉ coi là lưới an toàn**, không phải giải pháp chính, vì nó chỉ xóa chữ sai chứ không lấy lại lời thật bị mất. |
| S5 | Chặn kéo giãn: không chia cue dài theo tỉ lệ ký tự khi mốc thời gian không đáng tin; dùng mốc theo từ (`word_timestamps`) hoặc giới hạn thời lượng tối đa của một cue. |

**Dùng lời bài hát có sẵn (nếu người dùng cung cấp, tùy chọn):**
- Ưu tiên **"bám lời" (lyric snapping)** trước: giữ mốc thời gian từ Whisper, so khớp văn bản Whisper với lời gốc (so khớp chuỗi mờ/biên tập khoảng cách), thay bằng dòng lời gốc tương ứng khi độ giống đủ cao; dòng không khớp được thì đánh dấu để người dùng xem. Cách này không phụ thuộc ngôn ngữ nên hợp với lời trộn Anh/Pidgin/Việt.
- **Căn lời cưỡng bức (forced alignment)** (ví dụ qua wav2vec2/WhisperX) cho mốc thời gian chính xác hơn nhưng thường cần mô hình căn **theo từng ngôn ngữ**; với lời trộn ngôn ngữ và tiếng Pidgin có thể không có mô hình phù hợp. Chỉ thử sau khi tra cứu hỗ trợ ngôn ngữ thật sự, và ghi rõ giới hạn.
- Nếu lời cung cấp có phần ngoặc đơn/chú thích (ví dụ phần đệm giọng), cho cấu hình quy định xử lý (giữ/bỏ).

**Câu hỏi cần người dùng trả lời trước khi chọn hướng:** nội dung chính của họ là **bài hát/MV** hay **lời thoại thường**? Hai trường hợp cần mặc định rất khác nhau (`audio_mode`, tách giọng, bám lời). Nếu cả hai, thêm lựa chọn "chế độ nội dung" trong UI thay vì một bộ mặc định cho tất cả.

## 4. Quy tắc quyết định

- Giữ một thay đổi nếu WER/CER **giảm có ý nghĩa** trên clip thử **và** không làm tăng bịa chữ/lặp ở đoạn im lặng. Cải thiện nhỏ nằm trong dao động đo thì **không** coi là bằng chứng; nếu có thể, thử thêm một clip thứ hai trước khi kết luận.
- Không đưa một giá trị mặc định mới từ chỉ một clip. Ghi rõ "kết luận từ N clip".
- Nếu hai cấu hình ngang nhau, chọn cái đơn giản và rẻ hơn.

## 5. Sửa tạm thời (áp dụng trước, rủi ro thấp, dễ hoàn tác)

Chỉ làm các việc này **nếu thí nghiệm xác nhận hoặc trong lúc chờ thí nghiệm** và luôn cho phép tắt qua cấu hình:

1. **Đổi mặc định `audio_mode` thành `speech`** cho nội dung hội thoại; giữ `music` làm lựa chọn. (Kiểm tra E1 trước khi đổi cố định.) Hiển thị rõ lựa chọn trong UI kèm chú thích khi nào dùng.
2. **Giữ ngôn ngữ từng đoạn** thay vì ghi đè bằng ngôn ngữ chung trong `pipeline.py`; chuyển đúng ngôn ngữ từng lô cho translator.
3. **Đổi cách chia đoạn:** thay cố định 25s bằng chia tại khoảng lặng (có gối đầu vài giây và loại trùng ở mối nối), hoặc cho Whisper xử lý cả file khi E2 cho thấy tốt hơn. Phải có test mối nối không mất/lặp chữ.
4. **Thêm tùy chọn `initial_prompt`** (tên riêng/thuật ngữ) vào cấu hình và UI.
5. **Thêm bước tiền xử lý âm thanh tùy chọn** (`loudnorm` + khử ồn nhẹ), mặc định tắt, bật khi E4 chứng minh có ích.
6. **Thêm tách giọng khỏi nhạc như tùy chọn** (mặc định tắt) nếu E3 cho thấy lợi ích rõ.
7. **Ghi thêm vào báo cáo (`subtitle_report.json`):** ngôn ngữ từng đoạn, số đoạn nghi bịa chữ (đoạn không có tiếng nói theo VAD nhưng có chữ), các tham số ASR đã dùng.
8. **Luôn giữ `.srt` gốc** cạnh bản dịch (đã có), để người dùng so và sửa tay.
9. **Cơ chế can thiệp thủ công (Human-in-the-loop / Chỉnh sửa & xem trước SRT trước khi dịch/ghép):**
   - **Xem trước & biên tập (Preview & Edit):** Trong giao diện (WebUI/CLI/Colab), cho phép hiển thị nội dung `.srt` gốc ngay sau bước ASR để người dùng đọc soát lỗi, trực tiếp sửa các từ nghe nhầm/bịa chữ hoặc dán đè file SRT đã tinh chỉnh.
   - **Chế độ tách rời (Two-stage execution / Custom SRT input):** `SubtitlePipeline` hỗ trợ nhận đường dẫn file `.srt` có sẵn (`custom_srt` hoặc `skip_asr=True`), bỏ qua bước Whisper và đi thẳng vào phân đoạn -> dịch thuật (Gemini) -> ghép sub (Burn-in). Đây là giải pháp phụ cực kỳ an toàn và linh hoạt cho các clip khó/bài hát phức tạp.

## 6. Bước sửa lỗi nhận dạng bằng LLM (chỉ sau khi 3.x xong)

- Nếu sau các sửa trên vẫn còn lỗi nghe nhầm rõ ràng, thêm một lượt tùy chọn: LLM đọc `.srt` gốc + ngữ cảnh/bảng thuật ngữ và **chỉ sửa chỗ chắc chắn sai**, đánh dấu chỗ không chắc, không viết lại lời thoại, giữ nguyên số đoạn và mốc thời gian. So sánh WER trước/sau để chứng minh có ích; nếu làm tệ hơn thì bỏ.
- Tuân theo quy tắc nội dung nhạy cảm đã nêu ở kế hoạch phụ đề: **không** dùng prompt đánh lừa để vượt bộ lọc của model; xử lý bị chặn bằng cách thử lại/dự phòng/giữ bản gốc kèm cờ.

## 7. Ràng buộc và an toàn

- Không commit video, âm thanh, khóa API, trọng số model, hoặc file lớn. Thư mục `eval/` phải nằm trong `.gitignore`.
- Khóa API đọc từ biến môi trường hoặc Colab Secrets; không ghi vào log hay notebook.
- Giải phóng model giữa các bước (VRAM): Whisper, model tách giọng và SeedVR2 không chạy cùng lúc.
- Mọi thay đổi mã đi kèm test (dùng backend/ASR giả, không gọi model hay API thật trong `pytest`).
- Mọi con số (WER/CER, thời gian, VRAM) phải đến từ lần chạy thật; nếu không chạy được trên Colab thì ghi "chưa đo".

## 8. Báo cáo cần nộp cho người dùng

Viết `docs/asr_diagnosis.md` gồm:
1. Phân loại lỗi (3.1) và kết luận lỗi nằm ở ASR hay ở dịch.
2. Bảng kết quả các thí nghiệm (cấu hình, WER/CER, thời gian, VRAM) và nhận xét.
3. Giả thuyết nào được **xác nhận / bác bỏ / chưa kết luận**, kèm bằng chứng.
4. Những sửa tạm đã áp dụng, giá trị mặc định mới và lý do.
5. Việc còn tồn tại và gợi ý bước tiếp theo.

Sau báo cáo, **dừng lại hỏi người dùng** trước khi đổi mặc định ảnh hưởng nhiều đến mọi video, hoặc trước khi thêm phụ thuộc nặng (tách giọng).
