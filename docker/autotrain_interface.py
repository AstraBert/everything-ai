import subprocess as sp
import gradio as gr
import subprocess as sp


def build_command(hf_usr, hf_token, configpath):
    sp.run(f"export HF_USERNAME=\"{hf_usr}\"", shell=True)
    sp.run(f"export HF_TOKEN=\"{hf_token}\"", shell=True)
    sp.run(f"autotrain --config {configpath}", shell=True)
    return f"export HF_USERNAME={hf_usr}\nexport HF_TOKEN={hf_token}\nautotrain --config {configpath}"
    

demo = gr.Interface(
    build_command,
    [
        gr.Textbox(
            label="HF username",
            info="Your HF username",
            lines=3,
            value=f"your-cute-name",
        ),
        gr.Textbox(
            label="HF write token",
            info="An HF token that has write permissions on your repository",
            lines=3,
            value=f"your-powerful-token",
        ),
        gr.Textbox(
            label="Yaml configuration file",
            info="Path to the yaml configuration file containing the information to use autotrain",
            lines=3,
            value="/path/to/config.yaml",
        )
    ],
    title="everything-ai-autotrain",
    outputs="textbox",
    theme=gr.themes.Base()
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

	