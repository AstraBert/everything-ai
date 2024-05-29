from utils import Translation, PDFdatabase, NeuralSearcher
import gradio as gr
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

argparse = ArgumentParser()
argparse.add_argument(
    "-m",
    "--model",
    help="HuggingFace Model identifier, such as 'google/flan-t5-base'",
    required=True,
)

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


mod = args.model
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(mod).to(device)
tokenizer = AutoTokenizer.from_pretrained(mod)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, repetition_penalty=1.2, temperature=0.4)

def reply(message, history):
    global pdfdb
    txt = Translation(message, "en")
    if txt.original == "en" and lan.replace("\\","").replace("'","") == "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        results = txt2txt.search(message)
        response = pipe(results[0]["text"])
        return response[0]["generated_text"]
    elif txt.original == "en" and lan.replace("\\","").replace("'","") != "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0]["text"], txt.original)
        res = t.translatef()
        response = pipe(res)
        return response[0]["generated_text"]
    elif txt.original != "en" and lan.replace("\\","").replace("'","") == "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        results = txt2txt.search(message)
        transl = Translation(results[0]["text"], "en")
        translation = transl.translatef()
        response = pipe(translation)
        t = Translation(response[0]["generated_text"], txt.original)
        res = t.translatef()
        return res
    else:
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0]["text"], txt.original)
        res = t.translatef()
        response = pipe(res)
        tr = Translation(response[0]["generated_text"], txt.original)
        ress = tr.translatef()
        return ress 
    
demo = gr.ChatInterface(fn=reply, title="everything-ai-retrievaltext")
demo.launch(server_name="0.0.0.0", share=False)