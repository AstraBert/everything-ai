import gradio as gr
from utils import Translation



def reply(message, history):
    txt = Translation(message, "en")
    if txt.original == "en":
        image = f"https://pollinations.ai/p/{message.replace(' ', '_')}"
        return f"Here's your image:\n![generated_image]({image})"
    else:
        translation = txt.translatef()
        image = f"https://pollinations.ai/p/{translation.replace(' ', '_')}"
        t = Translation("Here's your image:", txt.original)
        res = t.translatef()
        return f"{res}:\n![generated_image]({image})"


demo = gr.ChatInterface(fn=reply, title="everything-ai-pollinations-imgs")
demo.launch(server_name="0.0.0.0", share=False)