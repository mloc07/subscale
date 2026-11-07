"""Test namprice227/F5-TTS-Vietnamese1000h-Pretrain checkpoint."""

import os
import subprocess
import time
import torch

def test_namprice():
    work_dir = "/content/phase0_test"
    ref_wav = f"{work_dir}/ref_peter_griffin.wav"
    out_wav = f"{work_dir}/dub_namprice_peter.wav"
    
    from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
    from huggingface_hub import hf_hub_download
    
    repo_id = "namprice227/F5-TTS-Vietnamese1000h-Pretrain"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model_finetune_200h.pt")
    vocab_path = hf_hub_download(repo_id=repo_id, filename="finetune_vocab.txt")
    print(f"Ckpt: {ckpt_path}")
    print(f"Vocab: {vocab_path}")
    
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
        vocab_file=vocab_path,
        device=device
    )
    print("namprice227 model loaded successfully!")
    
    ref_text = "How do you know what's inside you unless you test yourself?"
    gen_text = "Làm sao biết được mình có gì nếu không thử sức?"
    
    print("Inference with namprice227 checkpoint...")
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
    print(f"Generated namprice voice: {out_wav}, duration: {len(wav)/sr:.2f}s")
    
    # ASR eval
    from test_phase0_audio import preload_cuda12_libs, load_wav_float32
    preload_cuda12_libs()
    audio_arr = load_wav_float32(out_wav)
    from faster_whisper import WhisperModel
    asr_model = WhisperModel("base", device="cuda", compute_type="float16")
    segs, info = asr_model.transcribe(audio_arr, language="vi")
    rec = " ".join([s.text.strip() for s in segs])
    print(f"ASR of namprice audio: '{rec}' (Lang prob: {info.language_probability:.2f})")

if __name__ == "__main__":
    test_namprice()
