from transformers import pipeline
from argparse import ArgumentParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from utils import merge_pdfs
import gradio as gr
import time
import torch

histr = [[None, "Hi, I'm **everything-ai-summarization**🤖.\nI'm here to assist you and let you summarize _your_ texts and _your_ pdfs!\nCheck [my website](https://astrabert.github.io/everything-ai/) for troubleshooting and documentation reference\nHave fun!😊"]]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer = pipeline("summarization", model=model_checkpoint, device=device)

def convert_none_to_str(l: list):
    newlist = []
    for i in range(len(l)):
        if l[i] is None or type(l[i])==tuple:
            newlist.append("")
        else:
            newlist.append(l[i])
    return tuple(newlist)

def pdf2string(pdfpath):
    loader = PyPDFLoader(pdfpath)
    documents = loader.load()

    ### Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    fulltext = ""
    for text in texts:
        fulltext += text.page_content+"\n\n\n"
    return fulltext

def add_message(history, message):
    global histr
    if history is not None:
        if len(message["files"]) > 0:
            history.append((message["files"], None))
            histr.append([message["files"], None])
        if message["text"] is not None and message["text"] != "":
            history.append((message["text"], None))
            histr.append([message["text"], None])
    else:
        history = histr
        add_message(history, message)
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history):
    global histr
    if not history is None:
        if type(history[-1][0]) != tuple:
            text = history[-1][0]
            response = summarizer(text, max_length=int(len(text.split(" "))*0.5), min_length=int(len(text.split(" "))*0.05), do_sample=False)[0]
            response = response["summary_text"]
            histr[-1][1] = response
            history[-1][1] = ""
            for character in response:
                history[-1][1] += character
                time.sleep(0.05)
                yield history
        if type(history[-1][0]) == tuple:
            filelist = []
            for i in history[-1][0]:
                filelist.append(i)
            finalpdf = merge_pdfs(filelist)
            text = pdf2string(finalpdf)
            response = summarizer(text, max_length=int(len(text.split(" "))*0.5), min_length=int(len(text.split(" "))*0.05), do_sample=False)[0]
            response = response["summary_text"]
            histr[-1][1] = response
            history[-1][1] = ""
            for character in response:
                history[-1][1] += character
                time.sleep(0.05)
                yield history
    else:
        history = histr
        bot(history)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [[None, "Hi, I'm **everything-ai-summarization**🤖.\nI'm here to assist you and let you summarize _your_ texts and _your_ pdfs!\nCheck [my website](https://astrabert.github.io/everything-ai/) for troubleshooting and documentation reference\nHave fun!😊"]],
        label="everything-rag",
        elem_id="chatbot",
        bubble_full_width=False,
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["pdf"], placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)

	