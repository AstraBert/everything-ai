import gradio as gr
from utils import Translation, NeuralSearcheR
from gradio_client import Client
import os
import vecs
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser

argparse = ArgumentParser()

argparse.add_argument(
    "-gc",
    "--gradio_client",
    help="Spaces API to connect with",
    required=True,
)

argparse.add_argument(
    "-sdu",
    "--supabase_database_url",
    help="URL for Supabase database",
    required=True
)

argparse.add_argument(
    "-cn",
    "--collection_name",
    help="Name of the Supabase collection",
    required=True
)

argparse.add_argument(
    "-l",
    "--language",
    help="Language of the written content contained in the pdfs",
    required=False,
    default="en"
)

argparse.add_argument(
    "-en",
    "--encoder",
    help="Encoder used in text vectorization",
    required=False,
    default="all-MiniLM-L6-v2"
)

argparse.add_argument(
    "-s",
    "--size",
    help="Size of the vectors",
    required=False,
    default=384,
    type=int
)

args = argparse.parse_args()


gradcli = args.gradio_client
supdb = args.supabase_database_url
collname = args.collection_name
lan = args.language
encd = args.encoder
sz = args.size


collection_name = collname
encoder = SentenceTransformer(encd)
client = supdb
api_client = Client(gradcli)
lan = "en"
vx = vecs.create_client(client)
docs = vx.get_or_create_collection(name=collection_name, dimension=sz)

def reply(message, history):
    global docs
    global encoder
    global api_client
    global lan
    txt = Translation(message, "en")
    print(txt.original, lan)
    if txt.original == "en" and lan == "en":
        txt2txt = NeuralSearcheR(docs, encoder)
        results = txt2txt.search(message)
        response = api_client.predict(
            f"Context: {results[0][2]['Content']}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        return response
    elif txt.original == "en" and lan != "en":
        txt2txt = NeuralSearcheR(docs, encoder)
        transl = Translation(message, lan)
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0][2]['Content'], txt.original)
        res = t.translatef()
        response = api_client.predict(
            f"Context: {res}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        response = Translation(response, txt.original)
        return response.translatef()
    elif txt.original != "en" and lan == "en":
        txt2txt = NeuralSearcheR(docs, encoder)
        results = txt2txt.search(message)
        transl = Translation(results[0][2]['Content'], "en")
        translation = transl.translatef()
        response = api_client.predict(
            f"Context: {translation}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        t = Translation(response, txt.original)
        res = t.translatef()
        return res
    else:
        txt2txt = NeuralSearcheR(docs, encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0][2]['Content'], txt.original)
        res = t.translatef()
        response = api_client.predict(
            f"Context: {res}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        tr = Translation(response, txt.original)
        ress = tr.translatef()
        return ress


demo = gr.ChatInterface(fn=reply, title="everything-ai-supabase2spacesapi")
demo.launch(server_name="0.0.0.0", share=False)