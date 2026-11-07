"""Check Hugging Face model repository files on Colab."""

from huggingface_hub import list_repo_files

repos = [
    'hynt/F5-TTS-Vietnamese-ViVoice',
    'danhtran2mind/Vi-F5-TTS',
    'namprice227/F5-TTS-Vietnamese1000h-Pretrain',
    'SWivid/F5-TTS'
]

for repo in repos:
    try:
        files = list(list_repo_files(repo))
        print(f"=== {repo} ===")
        for f in files:
            if any(f.endswith(ext) for ext in ('.pt', '.safetensors', '.bin', '.json', '.yaml', '.txt')):
                print("  ", f)
    except Exception as e:
        print(f"=== {repo} === Error: {e}")
