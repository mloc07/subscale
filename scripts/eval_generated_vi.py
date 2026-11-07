"""Evaluate generated Vietnamese cloned speech using faster-whisper ASR."""

import os
import wave
import numpy as np

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
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    if framerate != 16000:
        import scipy.signal
        num_samples = int(len(audio) * 16000 / framerate)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio

def evaluate():
    from test_phase0_audio import preload_cuda12_libs
    preload_cuda12_libs()
    
    dub_wav = "/content/phase0_test/dub_vietnamese_peter.wav"
    audio_arr = load_wav_float32(dub_wav)
    
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio_arr, language="vi", beam_size=5)
    
    recognized_text = " ".join([s.text.strip() for s in segments])
    expected_text = "Làm sao biết được mình có gì nếu không thử sức"
    
    print(f"Recognized (ASR): '{recognized_text}'")
    print(f"Expected (Text):   '{expected_text}'")
    print(f"Language detected: {info.language} ({info.language_probability:.2f})")

if __name__ == "__main__":
    evaluate()
