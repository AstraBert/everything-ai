from transformers import pipeline
from argparse import ArgumentParser
import torch
import gradio as gr
import numpy as np


argparse = ArgumentParser()
argparse.add_argument(
    "-m",
    "--model",
    help="HuggingFace Model identifier, such as 'google/flan-t5-base'",
    required=True,
)

args = argparse.parse_args()


mod = args.model
mod = mod.replace("\"", "").replace("'", "")

model_checkpoint = mod

# Audio class
classifier = pipeline(task="automatic-speech-recognition", model=mod)

def classify_text(audio):
    global classifier
    sr, data = audio
    short_tensor = data.astype(np.float32)
    res = classifier(short_tensor)
    return res["text"]

input_audio = gr.Audio(
    sources=["upload","microphone"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)
demo = gr.Interface(
    title="everything-ai-speechrec",
    fn=classify_text,
    inputs=input_audio,
    outputs="text"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
