import torch
from transformers import pipeline
from PIL import Image
from argparse import ArgumentParser

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

pipe = pipeline("image-to-text", model=model_checkpoint)

def get_results(image, ppln=pipe):
    img = Image.fromarray(image)
    result = ppln(img, prompt="", generate_kwargs={"max_new_tokens": 1024})
    return result[0]["generated_text"].capitalize()

import gradio as gr
## Build interface with loaded image + ouput from the model
demo = gr.Interface(get_results, gr.Image(), "text")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)

