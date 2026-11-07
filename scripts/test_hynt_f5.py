"""Test hynt/F5-TTS-Vietnamese-ViVoice checkpoint."""

import os
import subprocess
import time
import torch

def test_hynt():
    work_dir = "/content/phase0_test"
    ref_wav = f"{work_dir}/ref_peter_griffin.wav"
    out_wav = f"{work_dir}/dub_hynt_peter.wav"
    
    from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
    from huggingface_hub import hf_hub_download
    
    repo_id = "hynt/F5-TTS-Vietnamese-ViVoice"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model_last.pt")
    
    # Check if there is vocab in this repo or default vocab
    from huggingface_hub import list_repo_files
    files = list(list_repo_files(repo_id))
    print(f"Files in {repo_id}: {files}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocoder = load_vocoder(vocoder_name="vocos", device=device)
    
    from f5_tts.model import DiT
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model(
        model_cls,
        model_cfg,
        ckpt_path,
        mel_spec_type="vocos",
        device=device
    )
    
    ref_text = "How do you know what's inside you unless you test yourself?"
    gen_text = "Làm sao biết được mình có gì nếu không thử sức?"
    
    print("Inference with hynt checkpoint...")
    wav, sr, spec = infer_process(
        ref_audio=ref_wav,
        ref_text=ref_text,
        gen_text=gen_text,
        model_obj=model,
        vocoder=vocoder,
        mel_spec_type="vocos",
        speed=1.0,
        device=device
    )
    
    import soundfile as sf
    sf.write(out_wav, wav, sr)
    print(f"Generated hynt voice: {out_wav}, duration: {len(wav)/sr:.2f}s")
    
    # ASR eval
    from test_phase0_audio import preload_cuda12_libs, load_wav_float32
    preload_cuda12_libs()
    audio_arr = load_wav_float32(out_wav)
    from faster_whisper import WhisperModel
    asr_model = WhisperModel("base", device="cuda", compute_type="float16")
    segs, info = asr_model.transcribe(audio_arr, language="vi")
    rec = " ".join([s.text.strip() for s in segs])
    print(f"ASR of hynt audio: '{rec}' (Lang prob: {info.language_probability:.2f})")

if __name__ == "__main__":
    test_hynt()
