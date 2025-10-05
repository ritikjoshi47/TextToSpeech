import requests
import os

# Create directory if needed
os.makedirs('models', exist_ok=True)

# File details from Hugging Face (rhasspy/piper-voices)
files = [
    {
        'name': 'en_US-lessac-medium.onnx',
        'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx',
        'path': 'models/en_US-lessac-medium.onnx'
    },
    {
        'name': 'en_US-lessac-medium.onnx.json',
        'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json',
        'path': 'models/en_US-lessac-medium.onnx.json'
    }
]

for file_info in files:
    print(f"Downloading {file_info['name']}...")
    response = requests.get(file_info['url'], stream=True)
    response.raise_for_status()
    
    with open(file_info['path'], 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Saved {file_info['name']} ({os.path.getsize(file_info['path'])} bytes)")

print("Downloads complete! Add 'models/' folder to your GitHub repo with LFS.")
