import subprocess as sp
import gradio as gr

TASK_TO_SCRIPT = {"retrieval-text-generation": "retrieval_text_generation.py", "agnostic-text-generation": "agnostic_text_generation.py", "text-summarization": "text_summarization.py", "image-generation": "image_generation.py", "image-generation-pollinations": "image_generation_pollinations.py", "image-classification": "image_classification.py", "image-to-text": "image_to_text.py", "retrieval-image-search": "retrieval_image_search.py"}


def build_command(tsk, mod="None", pdff="None", dirs="None", lan="None", imdim="512"):
    if tsk != "retrieval-text-generation" and tsk != "image-generation-pollinations" and tsk != "retrieval-image-search":
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod}", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod}"
    elif tsk == "retrieval-text-generation":
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod} -pf '{pdff}' -d '{dirs}' -l '{lan}'", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod} -pf '{pdff}' -d '{dirs}' -l '{lan}'"
    elif tsk == "image-generation-pollinations":
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]}", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]}"
    else:
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -d {dirs} -id {imdim} -m {mod}", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]} -d {dirs} -id {imdim} -m {mod}"

demo = gr.Interface(
    build_command,
    [
        gr.Textbox(
            label="Task",
            info="Task you want your assistant to help you with",
            lines=3,
            value=f"Choose one of the following: {','.join(list(TASK_TO_SCRIPT.keys()))}; if you choose 'image-generation-pollinations', you do not need to specify anything else",
        ),
        gr.Textbox(
            label="Model",
            info="AI model you want your assistant to run with",
            lines=3,
            value="None",
        ),
        gr.Textbox(
            label="PDF file(s)",
            info="Single pdf file or N pdfs reported like this: /path/to/file1.pdf,/path/to/file2.pdf,...,/path/to/fileN.pdf (there is no strict naming, you just need to provide them comma-separated): only available with 'retrieval-text-generation'",
            lines=3,
            value="None",
        ),
        gr.Textbox(
            label="Directory",
            info="Directory where all your pdfs or images (.jpg, .jpeg, .png) of interest are stored (only available with 'retrieval-text-generation' for pdfs and 'retrieval-image-search' for images)",
            lines=3,
            value="None",
        ),
        gr.Textbox(
            label="Language",
            info="Language of the written content contained in the pdfs",
            lines=3,
            value="None",
        ),
        gr.Textbox(
            label="Image dimension",
            info="Dimension of the image (this is generally model and/or task-dependent!)",
            lines=3,
            value=f"e.g.: 512, 384, 758...",
        ),
    ],
    outputs="textbox",
    theme=gr.themes.Base()
)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8760, share=False)

	