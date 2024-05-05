import gradio as gr
from utils import Translation
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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

model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, repetition_penalty=1.2, temperature=0.4)


def reply(message, history):
    txt = Translation(message, "en")
    if txt.original == "en":
        response = pipe(message)
        return response[0]["generated_text"]
    else:
        translation = txt.translatef()
        response = pipe(translation)
        t = Translation(response[0]["generated_text"], txt.original)
        res = t.translatef()
        return res


demo = gr.ChatInterface(fn=reply, title="Multilingual-Bloom Bot")
demo.launch(server_name="0.0.0.0", share=False)