from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr
from argparse import ArgumentParser
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from utils import *
import os
import subprocess as sp

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

sp.run("rm -rf memory.db", shell=True)

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

NAME2CHAT = {"Cohere": ChatCohere, "claude-3-opus-20240229": ChatAnthropic, "claude-3-sonnet-20240229": ChatAnthropic, "claude-3-haiku-20240307": ChatAnthropic, "llama3-8b-8192": ChatGroq, "llama3-70b-8192": ChatGroq, "mixtral-8x7b-32768": ChatGroq, "gemma-7b-it": ChatGroq, "gpt-4o": ChatOpenAI, "gpt-3.5-turbo-0125": ChatOpenAI}
NAME2APIKEY = {"Cohere": "COHERE_API_KEY", "claude-3-opus-20240229": "ANTHROPIC_API_KEY", "claude-3-sonnet-20240229": "ANTHROPIC_API_KEY", "claude-3-haiku-20240307": "ANTHROPIC_API_KEY", "llama3-8b-8192": "GROQ_API_KEY", "llama3-70b-8192": "GROQ_API_KEY", "mixtral-8x7b-32768": "GROQ_API_KEY", "gemma-7b-it": "GROQ_API_KEY", "gpt-4o": "OPENAI_API_KEY", "gpt-3.5-turbo-0125": "OPENAI_API_KEY"}



system_template = "You are an helpful assistant that can rely on this: {context} and on the previous message history as context, and from that you build a context and history-aware reply to this user input:"



def reply(message, history, name, api_key, temperature, max_new_tokens, sessionid):
    global pdfdb
    os.environ[NAME2APIKEY[name]]  = api_key
    if name == "Cohere":
        model = NAME2CHAT[name](temperature=temperature, max_tokens=max_new_tokens)
    else:
        model = NAME2CHAT[name](model=name,temperature=temperature, max_tokens=max_new_tokens)
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")]
    )
    chain = prompt_template | model
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    txt = Translation(message, "en")
    if txt.original == "en" and lan.replace("\\","").replace("'","") == "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        results = txt2txt.search(message)
        response = runnable_with_history.invoke({"context": results[0]["text"], "input": message}, config={"configurable": {"session_id": sessionid}})##CONFIGURE!
        return response.content
    elif txt.original == "en" and lan.replace("\\","").replace("'","") != "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0]["text"], txt.original)
        res = t.translatef()
        response = runnable_with_history.invoke({"context": res, "input": message}, config={"configurable": {"session_id": sessionid}})##CONFIGURE!
        return response.content
    elif txt.original != "en" and lan.replace("\\","").replace("'","") == "None":
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        results = txt2txt.search(message)
        transl = Translation(results[0]["text"], "en")
        translation = transl.translatef()
        response = runnable_with_history.invoke({"context": translation, "input": message}, config={"configurable": {"session_id": sessionid}})##CONFIGURE!
        t = Translation(response.content, txt.original)
        res = t.translatef()
        return res
    else:
        txt2txt = NeuralSearcher(pdfdb.collection_name, pdfdb.client, pdfdb.encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0]["text"], txt.original)
        res = t.translatef()
        response = runnable_with_history.invoke({"context": res, "input": message}, config={"configurable": {"session_id": sessionid}})##CONFIGURE!
        tr = Translation(response.content, txt.original)
        ress = tr.translatef()
        return ress 
    
chat_model = gr.Dropdown(
    [m for m in list(NAME2APIKEY)], label="Chat Model", info="Choose one of the available chat models"
    )

user_api_key = gr.Textbox(
    label="API key",
    info="Paste your API key here",
    lines=1,
    type="password",
)

user_temperature = gr.Slider(0, 1, value=0.5, label="Temperature", info="Select model temperature")

user_max_new_tokens = gr.Slider(0, 8192, value=1024, label="Max new tokens", info="Select max output tokens (higher number of tokens will result in a longer latency)")

user_session_id = gr.Textbox(label="Session ID",info="This alphanumeric code will link model reply to a specific message history of which the models will be aware when replying. Changing it will result in the loss of memory for your model",value="1")

additional_accordion = gr.Accordion(label="Parameters to be set before you start chatting", open=True)

demo = gr.ChatInterface(fn=reply, additional_inputs=[chat_model, user_api_key, user_temperature, user_max_new_tokens, user_session_id], additional_inputs_accordion=additional_accordion, title="everything-ai-buildyourllm")


if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", share=False)
