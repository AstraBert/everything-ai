from diffusers import DiffusionPipeline
import torch
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

pipe = DiffusionPipeline.from_pretrained(model_checkpoint, torch_dtype=torch.float32)

import gradio as gr
from utils import Translation



def reply(message, history):
    txt = Translation(message, "en")
    if txt.original == "en":
        image = pipe(message).images[0]
        image.save("generated_image.png")
        return "Here's your image:\n![generated_image](generated_image.png)"
    else:
        translation = txt.translatef()
        image = pipe(translation).images[0]
        image.save("generated_image.png")
        t = Translation("Here's your image:", txt.original)
        res = t.translatef()
        return f"{res}:\n![generated_image](generated_image.png)"


demo = gr.ChatInterface(fn=reply, title="everything-ai-sd-imgs")
demo.launch(server_name="0.0.0.0", share=False)