"""Test F5-TTS Vietnamese zero-shot voice cloning with exact repo checkpoint."""

import os
import subprocess
import time
import torch

def test_f5_vietnamese():
    work_dir = "/content/phase0_test"
    vocal_wav = f"{work_dir}/htdemucs/original_hq/vocals.wav"
    ref_wav = f"{work_dir}/ref_peter_griffin.wav"
    out_wav = f"{work_dir}/dub_vietnamese_peter.wav"
    
    # 1. Cut reference audio (2.35s to 6.35s -> 4 seconds of Peter Griffin voice)
    print("--- 1. Extracting Reference Vocal Segment ---")
    if not os.path.exists(ref_wav):
        cmd_cut = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", "2.35", "-to", "6.35",
            "-i", vocal_wav,
            "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", ref_wav
        ]
        subprocess.run(cmd_cut, check=True)
    print(f"Reference voice ready: {ref_wav}, size: {os.path.getsize(ref_wav)} bytes")
    
    # 2. Test F5-TTS Vietnamese Inference
    print("--- 2. Testing F5-TTS Vietnamese Inference ---")
    ref_text = "How do you know what's inside you unless you test yourself?"
    gen_text = "Làm sao biết được mình có gì nếu không thử sức?"
    
    from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
    from huggingface_hub import hf_hub_download
    
    repo_id = "danhtran2mind/Vi-F5-TTS"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="Vi_F5_TTS_ckpts/model_last.pt")
    vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.txt")
    print(f"Using ckpt: {ckpt_path}")
    print(f"Using vocab: {vocab_path}")

    from f5_tts.model import DiT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load vocoder
    print(f"Loading vocoder on {device}...")
    vocoder = load_vocoder(vocoder_name="vocos", device=device)
    
    # Load F5 model
    print(f"Loading DiT model on {device}...")
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
    print("Model loaded successfully into GPU memory!")
    
    # Run inference
    print("Generating Vietnamese speech cloned with Peter Griffin's voice...")
    t_gen0 = time.time()
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
    gen_time = time.time() - t_gen0
    
    import soundfile as sf
    sf.write(out_wav, wav, sr)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"Generated Vietnamese cloned speech in {gen_time:.2f}s!")
    print(f"Output saved: {out_wav} ({os.path.getsize(out_wav)} bytes), sr={sr}, duration={len(wav)/sr:.2f}s, Peak VRAM: {peak_vram:.2f} GB")

if __name__ == "__main__":
    test_f5_vietnamese()
