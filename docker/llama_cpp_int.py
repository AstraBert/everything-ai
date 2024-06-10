from utils import Translation, PDFdatabase, NeuralSearcher
import gradio as gr
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import os

argparse = ArgumentParser()

argparse.add_argument(
    "-pf",
    "--pdf_file",
    help="Single pdf file or N pdfs reported like this: /path/to/file1.pdf,/path/to/file2.pdf,...,/path/to/fileN.pdf (there is no strict naming, you just need to provide them comma-separated)",
    required=False,
    default="No file"
)

argparse.add_argument(
    "-d",
    "--directory",
    help="Directory where all your pdfs of interest are stored",
    required=False,
    default="No directory"
)

argparse.add_argument(
    "-l",
    "--language",
    help="Language of the written content contained in the pdfs",
    required=False,
    default="Same as query"
)

args = argparse.parse_args()


pdff = args.pdf_file
dirs = args.directory
lan = args.language


if pdff.replace("\\","").replace("'","") != "None" and dirs.replace("\\","").replace("'","") == "No directory":
    pdfs = pdff.replace("\\","/").replace("'","").split(",")
else:
    pdfs = [os.path.join(dirs.replace("\\","/").replace("'",""), f) for f in os.listdir(dirs.replace("\\","/").replace("'","")) if f.endswith(".pdf")]

client = QdrantClient(host="host.docker.internal", port="6333")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

pdfdb = PDFdatabase(pdfs, encoder, client)
pdfdb.preprocess()
pdfdb.collect_data()
pdfdb.qdrant_collection_and_upload()


import requests

def llama_cpp_respond(query, max_new_tokens):
    url = "http://localhost:8000/completion"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": query,
        "n_predict": int(max_new_tokens)
    }

    response = requests.post(url, headers=headers, json=data)

    a = response.json()
    return a["content"]


def reply(max_new_tokens, message):
    global pdfdb
    txt = Translation(message, "en")
    if txt.original == "en" and lan.replace("\\","").replace("'","") == "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        results = txt2txt.search(message)
        response = llama_cpp_respond(f"Context: {results[0]["text"]}, prompt: {message}", max_new_tokens)
        return response
    elif txt.original == "en" and lan.replace("\\","").replace("'","") != "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0]["text"], txt.original)
        res = t.translatef()
        response = llama_cpp_respond(f"Context: {res}, prompt: {message}", max_new_tokens)
        return response
    elif txt.original != "en" and lan.replace("\\","").replace("'","") == "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        results = txt2txt.search(message)
        transl = Translation(results[0]["text"], "en")
        translation = transl.translatef()
        response = llama_cpp_respond(f"Context: {translation}, prompt: {message}", max_new_tokens)
        t = Translation(response, txt.original)
        res = t.translatef()
        return res
    else:
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0]["text"], txt.original)
        res = t.translatef()
        response = llama_cpp_respond(f"Context: {res}, prompt: {message}", max_new_tokens)
        tr = Translation(response, txt.original)
        ress = tr.translatef()
        return ress 
    
demo = gr.Interface(
    reply,
    [
        gr.Textbox(
            label="Max new tokens",
            info="The number reported should not be higher than the one specified within the .env file",
            lines=3,
            value=f"512",
        ),
        gr.Textbox(
            label="Input query",
            info="Write your input query here",
            lines=3,
            value=f"What are penguins?",
        )
    ],
    title="everything-ai-llamacpp",
    outputs="textbox"
)
demo.launch(server_name="0.0.0.0", share=False)