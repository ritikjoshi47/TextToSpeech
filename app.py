import gradio as gr
import numpy as np
import torch
import soundfile as sf
from emotivoice import EmotiVoice  # From GitHub install
import os
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# EmotiVoice setup
MODEL_DIR = "models"
STYLE_ENCODER_PATH = os.path.join(MODEL_DIR, "style_encoder/best_ckpt.pt")
JOINT_TTS_PATH = os.path.join(MODEL_DIR, "prompt_tts_joint/best_ckpt.pt")
SIMBERT_DIR = os.path.join(MODEL_DIR, "simbert")

if not all(os.path.exists(p) for p in [STYLE_ENCODER_PATH, JOINT_TTS_PATH, SIMBERT_DIR]):
    raise FileNotFoundError("EmotiVoice models not found in models/ directory")

VOICE_OPTIONS = {
    "Female Seductive (Nicole-like)": "female_en_seductive",
    "Female Joyful": "female_en_joyful",
    "Male Calm": "male_en_calm"
}
DEFAULT_VOICE = "Female Seductive (Nicole-like)"

LANGUAGE_OPTIONS = ["English", "Chinese", "Auto"]
LANGUAGE_MAP = {"English": "en", "Chinese": "zh", "Auto": "auto"}

def tts_interface(text, voice_display, language_display):
    voice_name = VOICE_OPTIONS[voice_display]
    language = LANGUAGE_MAP[language_display]
    
    print(f"Text length: {len(text)}, Voice: {voice_name}, Language: {language}")

    # Initialize EmotiVoice
    emotivoice = EmotiVoice(
        style_encoder_path=STYLE_ENCODER_PATH,
        joint_tts_path=JOINT_TTS_PATH,
        simbert_dir=SIMBERT_DIR,
        device="cpu"  # Render CPU-only
    )
    
    # Generate with emotional prompt (e.g., append [sultry moan] if desired)
    if "[sultry moan]" in text.lower():
        text = text.replace("[sultry moan]", "")  # Process emotion internally
    
    audio_np, sample_rate = emotivoice.inference(text, voice=voice_name, lang=language)
    
    return (sample_rate, audio_np)

with gr.Blocks(theme=gr.themes.Soft(font=["Source Sans Pro", "Arial", "sans-serif"]), css=".gradio-container {max-width: none !important;}") as demo:
    gr.Markdown("# 🎤 EmotiVoice TTS - Emotional, No API Key!")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter naughty text with emotions, e.g., 'Oh baby, fuck me [sultry moan]' (500+ words OK)...",
                lines=10,
                max_lines=20
            )
            voice_select = gr.Dropdown(
                label="Select Voice",
                choices=list(VOICE_OPTIONS.keys()),
                value=DEFAULT_VOICE
            )
            language_select = gr.Dropdown(
                label="Select Language",
                choices=LANGUAGE_OPTIONS,
                value="Auto"
            )
            generate_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech", interactive=False)
    
    examples = gr.Examples(
        examples=[
            ["Oh darling, you're in for a treat. Let's dive into something wild and wicked [sultry moan]. Picture this: you're sprawled out... [full 500+ word text]", "Female Seductive (Nicole-like)", "English"],
        ],
        inputs=[text_input, voice_select, language_select],
        label="Examples"
    )

    generate_btn.click(
        fn=tts_interface,
        inputs=[text_input, voice_select, language_select],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch()
