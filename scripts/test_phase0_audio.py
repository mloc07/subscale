"""Phase 0 Audio Separation, ASR, and Gemini Translation Benchmark on Colab."""

import json
import time
import subprocess
import os
import wave
import numpy as np

def preload_cuda12_libs() -> None:
    import ctypes
    import glob
    import importlib.util

    if os.name != "posix":
        return
    for pkg, patterns in (("nvidia.cublas", ["libcublasLt.so.12", "libcublas.so.12"]),
                          ("nvidia.cudnn", ["libcudnn.so.9"])):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            spec = None
        if not spec or not spec.submodule_search_locations:
            continue
        lib_dir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        for pat in patterns:
            for f in glob.glob(os.path.join(lib_dir, pat)):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    print(f"Loaded {f}")
                except OSError as e:
                    print(f"Failed to load {f}: {e}")

def load_wav_float32(path):
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        
    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
        
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
        
    if framerate != 16000:
        import scipy.signal
        num_samples = int(len(audio) * 16000 / framerate)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
        
    return audio

def test_full_flow():
    import torch
    preload_cuda12_libs()
    
    video_path = "/content/tests_dub/FamilyGuy1_test_dub.mp4"
    work_dir = "/content/phase0_test"
    os.makedirs(work_dir, exist_ok=True)
    
    print("--- 1. Check Demucs Outputs ---")
    vocal_path = f"{work_dir}/htdemucs/original_hq/vocals.wav"
    no_vocal_path = f"{work_dir}/htdemucs/original_hq/no_vocals.wav"
    print(f"Vocals exists: {os.path.exists(vocal_path)} ({os.path.getsize(vocal_path)} bytes)")
    print(f"BGM exists: {os.path.exists(no_vocal_path)} ({os.path.getsize(no_vocal_path)} bytes)")
    
    print("--- 2. Faster-Whisper ASR via Audio Array ---")
    t0 = time.time()
    audio_arr = load_wav_float32(vocal_path)
    print(f"Loaded audio array: shape={audio_arr.shape}, max={np.max(np.abs(audio_arr)):.3f}")
    
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    segments, info = model.transcribe(audio_arr, beam_size=5, vad_filter=True)
    
    seg_list = []
    for s in segments:
        text = s.text.strip()
        if text:
            dur = round(float(s.end - s.start), 2)
            seg_list.append({
                "id": len(seg_list),
                "start": round(float(s.start), 2),
                "end": round(float(s.end), 2),
                "duration": dur,
                "text": text
            })
            
    asr_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"ASR done in {asr_time:.2f}s, Peak VRAM: {peak_vram:.2f} GB, Detected Lang: {info.language} ({info.language_probability:.2f})")
    print(f"Total segments: {len(seg_list)}")
    for s in seg_list[:8]:
        print(f"  [{s['start']}s -> {s['end']}s ({s['duration']}s)]: {s['text']}")
        
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("--- 3. Gemini Translation with Duration Budget ---")
    gemini_key = ""
    if os.path.exists("/content/.gemini_key"):
        with open("/content/.gemini_key") as f:
            gemini_key = f.read().strip()
            
    if not gemini_key:
        print("No gemini key found! Skip translation test.")
        return
        
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=gemini_key)
    
    prompt_items = []
    for s in seg_list[:8]:
        max_syllables = max(2, int(s["duration"] * 3.8))
        prompt_items.append({
            "id": s["id"],
            "original_en": s["text"],
            "duration_sec": s["duration"],
            "max_syllables_budget": max_syllables
        })
        
    system_instruction = (
        "Bạn là chuyên gia dịch thuật và biên kịch lồng tiếng phim hoạt hình chuyên nghiệp. "
        "Nhiệm vụ: Dịch lời thoại tiếng Anh sang tiếng Việt tự nhiên, hài hước, "
        "nhưng BẮT BUỘC tuân thủ NGÂN SÁCH THỜI LƯỢNG (max_syllables_budget). "
        "Số âm tiết câu tiếng Việt KHÔNG ĐƯỢC VƯỢT QUÁ max_syllables_budget để diễn viên lồng tiếng nói kịp trong khung hình. "
        "Trả về định dạng JSON array với các trường: id, vi_text, estimated_syllables."
    )
    
    t0 = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Dịch danh sách các câu thoại sau sang tiếng Việt lồng tiếng:\n{json.dumps(prompt_items, ensure_ascii=False, indent=2)}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.3
        )
    )
    trans_time = time.time() - t0
    print(f"Gemini 2.5 Flash translation took {trans_time:.2f}s:")
    try:
        translated_items = json.loads(response.text)
        for item in translated_items:
            orig = next((s for s in seg_list if s["id"] == item["id"]), None)
            dur = orig['duration'] if orig else '?'
            print(f"  [id {item['id']}|{dur}s] Gốc: \"{orig['text'] if orig else ''}\"")
            print(f"            Dịch: \"{item.get('vi_text')}\" (Số âm tiết: {item.get('estimated_syllables')})")
    except Exception as e:
        print("Raw response:", response.text)

if __name__ == "__main__":
    test_full_flow()
