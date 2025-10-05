import requests
import os
from huggingface_hub import hf_hub_download

os.makedirs('models', exist_ok=True)

# EmotiVoice checkpoints (from GitHub instructions)
files = [
    # Style encoder
    {
        'name': 'style_encoder.pth',
        'repo_id': 'syq163/outputs',
        'filename': 'style_encoder/ckpt/best_ckpt.pt',
        'path': 'models/style_encoder/best_ckpt.pt'
    },
    # Joint TTS model
    {
        'name': 'prompt_tts_joint.pth',
        'repo_id': 'syq163/outputs',
        'filename': 'prompt_tts_open_source_joint/ckpt/best_ckpt.pt',
        'path': 'models/prompt_tts_joint/best_ckpt.pt'
    },
    # SimBERT (Chinese embedding, optional but recommended)
    {
        'name': 'simbert_config.json',
        'url': 'https://huggingface.co/WangZeJun/simbert-base-chinese/resolve/main/config.json',
        'path': 'models/simbert/config.json'
    },
    {
        'name': 'simbert_pytorch_model.bin',
        'url': 'https://huggingface.co/WangZeJun/simbert-base-chinese/resolve/main/pytorch_model.bin',
        'path': 'models/simbert/pytorch_model.bin'
    },
    {
        'name': 'simbert_vocab.txt',
        'url': 'https://huggingface.co/WangZeJun/simbert-base-chinese/resolve/main/vocab.txt',
        'path': 'models/simbert/vocab.txt'
    }
]

for file_info in files:
    if 'url' in file_info:
        print(f"Downloading {file_info['name']} via URL...")
        response = requests.get(file_info['url'], stream=True)
        response.raise_for_status()
        with open(file_info['path'], 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Downloading {file_info['name']} via HF...")
        hf_hub_download(repo_id=file_info['repo_id'], filename=file_info['filename'], local_dir='models')
    print(f"Saved {file_info['name']} ({os.path.getsize(file_info['path'])} bytes)")

print("Downloads complete! Add 'models/' to your GitHub repo with LFS.")
