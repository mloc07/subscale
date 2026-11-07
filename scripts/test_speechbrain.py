"""Test SpeechBrain Speaker Embedding on Colab without gated token."""

import os
import subprocess
import time
import torch
import numpy as np

def test_speaker_embedding():
    work_dir = "/content/phase0_test"
    os.makedirs(work_dir, exist_ok=True)
    
    # Let's extract 2 different segments from FamilyGuy1 vocals:
    # Segment A: 2.35s to 6.35s (Peter Griffin)
    # Segment B: 6.35s to 9.35s (Peter Griffin)
    vocal_wav = f"{work_dir}/htdemucs/original_hq/vocals.wav"
    segA_wav = f"{work_dir}/segA_peter.wav"
    segB_wav = f"{work_dir}/segB_peter.wav"
    
    cmdA = ["ffmpeg", "-y", "-v", "error", "-ss", "2.35", "-to", "6.35", "-i", vocal_wav, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", segA_wav]
    cmdB = ["ffmpeg", "-y", "-v", "error", "-ss", "6.35", "-to", "9.35", "-i", vocal_wav, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", segB_wav]
    subprocess.run(cmdA, check=True)
    subprocess.run(cmdB, check=True)
    
    print("Testing SpeechBrain ECAPA-TDNN speaker embedding...")
    t0 = time.time()
    from speechbrain.inference.speaker import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    load_time = time.time() - t0
    print(f"SpeechBrain loaded in {load_time:.2f}s on {device}")
    
    # Compute embeddings
    import torchaudio
    signalA, fsA = torchaudio.load(segA_wav)
    signalB, fsB = torchaudio.load(segB_wav)
    
    embA = classifier.encode_batch(signalA.to(device)).squeeze().cpu().numpy()
    embB = classifier.encode_batch(signalB.to(device)).squeeze().cpu().numpy()
    
    # Cosine similarity
    cos_sim = np.dot(embA, embB) / (np.linalg.norm(embA) * np.linalg.norm(embB))
    print(f"Embedding shape: {embA.shape}")
    print(f"Cosine similarity between two Peter Griffin segments: {cos_sim:.4f}")
    
    # Also compare with a different audio (e.g. noise or TTS)
    tts_wav = f"{work_dir}/dub_vietnamese_peter.wav"
    if os.path.exists(tts_wav):
        signal_tts, fs_tts = torchaudio.load(tts_wav)
        if fs_tts != 16000:
            resampler = torchaudio.transforms.Resample(fs_tts, 16000).to(device)
            signal_tts = resampler(signal_tts.to(device)).cpu()
        emb_tts = classifier.encode_batch(signal_tts.to(device)).squeeze().cpu().numpy()
        sim_voice_clone = np.dot(embA, emb_tts) / (np.linalg.norm(embA) * np.linalg.norm(emb_tts))
        print(f"Cosine similarity between Original Peter and Cloned Voice: {sim_voice_clone:.4f}")

if __name__ == "__main__":
    test_speaker_embedding()
