import subprocess as sp
import gradio as gr

TASK_TO_SCRIPT = {"retrieval-text-generation": "retrieval_text_generation.py", "agnostic-text-generation": "agnostic_text_generation.py", "text-summarization": "text_summarization.py", "image-generation": "image_generation.py", "image-generation-pollinations": "image_generation_pollinations.py", "image-classification": "image_classification.py", "image-to-text": "image_to_text.py", "retrieval-image-search": "retrieval_image_search.py", "protein-folding": "protein_folding_with_esm.py", "video-generation": "video_generation.py", "speech-recognition": "speech_recognition.py", "spaces-api-supabase": "spaces_api_supabase.py", "audio-classification": "audio_classification.py", "autotrain": "autotrain_interface.py", "llama.cpp-and-qdrant": "llama_cpp_int.py", "build-your-llm": "build_your_llm.py"}


def build_command(tsk, mod="None", pdff="None", dirs="None", lan="None", imdim="512", gradioclient="None", supabaseurl="None", collectname="None", supenc="all-MiniLM-L6-v2", supdim="384"):
    if tsk != "retrieval-text-generation" and tsk != "image-generation-pollinations" and tsk != "retrieval-image-search" and tsk != "autotrain" and tsk != "protein-folding" and tsk != "spaces-api-supabase" and tsk != "llama.cpp-and-qdrant" and tsk!="build-your-llm":
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod}", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod}"
    elif tsk == "retrieval-text-generation":
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod} -pf '{pdff}' -d '{dirs}' -l '{lan}'", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]} -m {mod} -pf '{pdff}' -d '{dirs}' -l '{lan}'"
    elif tsk == "llama.cpp-and-qdrant" or tsk== "build-your-llm":
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -pf '{pdff}' -d '{dirs}' -l '{lan}'", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]} -pf '{pdff}' -d '{dirs}' -l '{lan}'"
    elif tsk == "image-generation-pollinations" or tsk == "autotrain" or tsk == "protein-folding":
        sp.run(f"python3 {TASK_TO_SCRIPT[tsk]}", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]}"
    elif tsk == "spaces-api-supabase":
        if lan == "None":
            sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -gc {gradioclient} -sdu {supabaseurl} -cn {collectname} -en {supenc} -s {supdim}", shell=True)
        else:
            sp.run(f"python3 {TASK_TO_SCRIPT[tsk]} -gc {gradioclient} -sdu {supabaseurl} -cn {collectname} -en {supenc} -s {supdim} -l {lan}", shell=True)
        return f"python3 {TASK_TO_SCRIPT[tsk]} -gc {gradioclient} -sdu {supabaseurl} -cn {collectname} -en {supenc} -s {supdim} -l {lan}"
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
            value=f"Choose one of the following: {','.join(list(TASK_TO_SCRIPT.keys()))}; if you choose 'image-generation-pollinations' or 'autotrain' or 'protein-folding', you do not need to specify anything else. If you choose 'spaces-api-supabase' you need to specify the Spaces API client, the database URL, the collection name, the Sentence-Transformers encoder used to upload the vectors to the Supabase database and the vectors size (optionally also the language)",
        ),
        gr.Textbox(
            label="Model",
            info="AI model you want your assistant to run with",
            lines=3,
            value="None",
        ),
        gr.Textbox(
            label="PDF file(s)",
            info="Single pdf file or N pdfs reported like this: /path/to/file1.pdf,/path/to/file2.pdf,...,/path/to/fileN.pdf (there is no strict naming, you just need to provide them comma-separated), please do not use '\\' as path separators: only available with 'retrieval-text-generation'",
            lines=3,
            value="No file",
        ),
        gr.Textbox(
            label="Directory",
            info="Directory where all your pdfs or images (.jpg, .jpeg, .png) of interest are stored (only available with 'retrieval-text-generation' for pdfs and 'retrieval-image-search' for images). Please do not use '\\' as path separators",
            lines=3,
            value="No directory",
        ),
        gr.Textbox(
            label="Language",
            info="Language of the written content contained in the pdfs",
            lines=1,
            value="None",
        ),
        gr.Textbox(
            label="Image dimension",
            info="Dimension of the image (this is generally model and/or task-dependent!)",
            lines=1,
            value=f"e.g.: 512, 384, 758...",
        ),
        gr.Textbox(
            label="Spaces API client",
            info="Client for Spaces API",
            lines=3,
            value=f"e.g.: eswardivi/Phi-3-mini-4k-instruct",
        ),
        gr.Textbox(
            label="Supabase Database URL",
            info="URL of the Supabase database (to use with Spaces API)",
            lines=3,
            value=f"e.g.: postgresql://postgres.reneogdbgdsbgdbgdsgbdlf:yourcomplexpasswordhere@aws-0-eu-central-1.pooler.supabase.com:5432/postgres",
        ),
        gr.Textbox(
            label="Supabase collection name",
            info="Name of the Supabase collectio (to use with Spaces API)",
            lines=2,
            value=f"e.g.: documents",
        ),
        gr.Textbox(
            label="Supabase Vector Encoder",
            info="Name of the sentence-transformers encoder you used to upload vectors to your supabase database",
            lines=2,
            value=f"e.g.: all-MiniLM-L6-v2",
        ),
        gr.Textbox(
            label="Supabase Vector Size",
            info="Size of vectors in you supabase database",
            lines=1,
            value=f"e.g.: 384",
        ),
    ],
    outputs="textbox",
    theme=gr.themes.Base()
)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8760, share=False)

	