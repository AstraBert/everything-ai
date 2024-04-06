import gradio as gr
import os
import time
from utils import *

vectordb = ""

def generate_welcome_message():
    return (None, "Hello! Welcome to the chatbot. You can enter a message or upload a file.")

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    if len(message["files"]) > 0:
        history.append((message["files"], None))
    if message["text"] is not None and message["text"] != "":
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history):
    global vectordb
    global tsk
    if type(history[-1][0]) != tuple:
        if vectordb == "":
            pipe = pipeline(tsk, tokenizer=tokenizer, model=model)
            response = pipe(history[-1][0])[0]
            response = response["generated_text"]
            history[-1][1] = ""
            for character in response:
                history[-1][1] += character
                time.sleep(0.05)
                yield history
        else:
            try:
                response = just_chatting(task=tsk, model=model, tokenizer=tokenizer, query=history[-1][0], vectordb=vectordb, chat_history=[convert_none_to_str(his) for his in history])["answer"]
                history[-1][1] = ""
                for character in response:
                    history[-1][1] += character
                    time.sleep(0.05)
                    yield history
            except Exception as e:
                response = f"Sorry, the error '{e}' occured while generating the response; check [troubleshooting documentation](https://astrabert.github.io/everything-rag/#troubleshooting) for more"
    if type(history[-1][0]) == tuple:
        filelist = []
        for i in history[-1][0]:
            filelist.append(i)
        finalpdf = merge_pdfs(filelist)
        vectordb = create_a_persistent_db(finalpdf, os.path.dirname(finalpdf)+"_localDB", os.path.dirname(finalpdf)+"_embcache")
        response = "VectorDB was successfully created, now you can ask me anything about the document you uploaded!😊"
        history[-1][1] = ""
        for character in response:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [[None, "Hi, I'm **everything-rag**🤖.\nI'm here to assist you and let you chat with _your_ pdfs!\nCheck [my website](https://astrabert.github.io/everything-rag/) for troubleshooting and documentation reference\nHave fun!😊"]],
        label="everything-rag",
        elem_id="chatbot",
        bubble_full_width=False,
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["pdf"], placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)
    clear = gr.ClearButton(chatbot)

demo.queue()
if __name__ == "__main__":
    demo.launch(share=False)

	