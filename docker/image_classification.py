from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from PIL import Image
from argparse import ArgumentParser
import torch

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageClassification.from_pretrained(model_checkpoint).to(device)
processor = AutoImageProcessor.from_pretrained(model_checkpoint)


pipe = pipeline("image-classification", model=model, image_processor=processor)

def get_results(image, ppln=pipe):
    img = Image.fromarray(image)
    result = ppln(img)
    scores = []
    labels = []
    for el in result:
        scores.append(el["score"])
        labels.append(el["label"])
    return labels[scores.index(max(scores))]

import gradio as gr
## Build interface with loaded image + ouput from the model
demo = gr.Interface(get_results, gr.Image(), "text")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)