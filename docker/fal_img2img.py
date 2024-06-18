import asyncio
import fal_client
import os
import gradio as gr
from PIL import Image

MAP_EXTS = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}

async def submit(image_path, prompt, seed):
    ext = image_path.split(".")[1]
    handler = await fal_client.submit_async(
        "comfy/astrabert/image2image",
        arguments={
            "ksampler_seed": seed,
            "cliptextencode_text": prompt,
            "image_load_image_path": f"data:image/{MAP_EXTS[ext]};base64,{image_path}"
        },
    )
    result = await handler.get()
    return result

def get_url(results):
    url = results['outputs'][list(results['outputs'].keys())[0]]['images'][0]['url']
    nm = results['outputs'][list(results['outputs'].keys())[0]]['images'][0]['filename']
    return f"![{nm}]({url})"


def render_image(api_key, image_path, prompt, seed):
    os.environ["FAL_KEY"] = api_key
    results = asyncio.run(submit(image_path, prompt, int(seed)))
    url = get_url(results)
    img = Image.open(image_path)
    return img, url


demo = gr.Interface(render_image, inputs=[gr.Textbox(label="API key", type="password", value="fal-******************"), gr.File(label="PNG/JPEG Image"), gr.Textbox(label="Prompt", info="Specify how you would like the image generation to be"),  gr.Textbox(label="Seed", info="Pass your seed here (if not interested, leave it as it is)", value="123498235498246")], outputs=[gr.Image(label="Your Base Image"), gr.Markdown(label="Generated Image")], title="everything-ai-img2img")

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)