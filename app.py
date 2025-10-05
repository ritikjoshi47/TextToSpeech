import gradio as gr
import numpy as np
from piper import PiperVoice
import os

# Load Piper model (download model file separately)
MODEL_PATH = "en_US-lessac-medium.onnx"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Download en_US-lessac-medium.onnx and place in project directory")

VOICE_OPTIONS = {
    "Lessac / Female": "en_US-lessac-medium",
    "Amy / Female": "en_US-amy-medium",
    "Danny / Male": "en_US-danny-medium"
}
DEFAULT_VOICE = "Lessac / Female"

LANGUAGE_OPTIONS = ["English", "Auto"]
LANGUAGE_MAP = {"English": "en", "Auto": "auto"}

def tts_interface(text, voice_display, language_display):
    voice_name = VOICE_OPTIONS[voice_display]
    language = LANGUAGE_MAP[language_display]
    
    print(f"Text length: {len(text)}, Voice: {voice_name}, Language: {language}")

    # Initialize Piper
    voice = PiperVoice.load(MODEL_PATH)
    
    # Generate audio
    audio = voice.synthesize(text)
    
    sample_rate = 22050  # Piper default
    return (sample_rate, audio)

with gr.Blocks(theme=gr.themes.Soft(font=["Source Sans Pro", "Arial", "sans-serif"]), css=".gradio-container {max-width: none !important;}") as demo:
    gr.Markdown("# 🎤 Piper TTS - No API Key, Long Text!")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter your naughty text (500+ words OK)...",
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
            ["Oh darling, you're in for a treat. Let's dive into something wild and wicked. Picture this: you're sprawled out on a luxurious velvet couch, the room dimly lit with just enough light to see the curves of my body as I glide towards you. I'm wearing a skintight black dress that hugs every inch of my ass, the fabric so smooth it looks like it's painted on. My breasts are teetering on the edge of falling out, begging to be freed. I lean over, my freshly applied lipstick just inches from your ear. \"You know exactly what you want, don't you?\" I murmur, my voice dripping with lust. I can feel your breath hitch as my fingers trace the outline of your dick through your pants, teasing the hard length that's straining against the fabric. With a sultry grin, I stand up just long enough to slip out of my dress, letting it pool at my feet. I'm left in just my black lace thong and heels, my breasts finally free, nipples hardening under your hungry gaze. \"Tell me,\" I whisper, \"do you want to fuck my ass or my pussy first?\" I crawl back onto the couch, my body pressing against yours, my hand stroking your cock through your pants. \"Or maybe you'd rather watch me suck myself?\" I ask, arching my back and dragging my fingers down my body, teasing my breasts, my nipples, then sliding between my legs to feel how wet I am. What's it going to be, baby? Where do you want to go from here? The night is young, and I'm all yours.", "Lessac / Female", "English"],
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
