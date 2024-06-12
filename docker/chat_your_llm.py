from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from utils import Translation
import time
import os
from langfuse.callback import CallbackHandler
import gradio as gr
import subprocess as sp


NAME2CHAT = {"Cohere": ChatCohere, "claude-3-opus-20240229": ChatAnthropic, "claude-3-sonnet-20240229": ChatAnthropic, "claude-3-haiku-20240307": ChatAnthropic, "llama3-8b-8192": ChatGroq, "llama3-70b-8192": ChatGroq, "mixtral-8x7b-32768": ChatGroq, "gemma-7b-it": ChatGroq, "gpt-4o": ChatOpenAI, "gpt-3.5-turbo-0125": ChatOpenAI}
NAME2APIKEY = {"Cohere": "COHERE_API_KEY", "claude-3-opus-20240229": "ANTHROPIC_API_KEY", "claude-3-sonnet-20240229": "ANTHROPIC_API_KEY", "claude-3-haiku-20240307": "ANTHROPIC_API_KEY", "llama3-8b-8192": "GROQ_API_KEY", "llama3-70b-8192": "GROQ_API_KEY", "mixtral-8x7b-32768": "GROQ_API_KEY", "gemma-7b-it": "GROQ_API_KEY", "gpt-4o": "OPENAI_API_KEY", "gpt-3.5-turbo-0125": "OPENAI_API_KEY"}

sp.run("rm -rf memory.db", shell=True)

def build_langfuse_handler(langfuse_host, langfuse_pkey, langfuse_skey):
    if langfuse_host!="None" and langfuse_pkey!="None" and langfuse_skey!="None":
        langfuse_handler = CallbackHandler(
            public_key=langfuse_pkey,
            secret_key=langfuse_skey,
            host=langfuse_host
        )
        return langfuse_handler, True
    else:
        return "No langfuse", False

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chatmemory.db")

def reply(message, history, name, api_key, temperature, max_new_tokens,langfuse_host, langfuse_pkey, langfuse_skey, system_template, sessionid):
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
    lf_handler, truth = build_langfuse_handler(langfuse_host, langfuse_pkey, langfuse_skey)
    chain = prompt_template | model
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    txt = Translation(message, "en")
    if txt.original == "en":
        if not truth:
            response = runnable_with_history.invoke({"input": message}, config={"configurable": {"session_id": sessionid}})##CONFIGURE!
        else:
            response = runnable_with_history.invoke({"input": message}, config={"configurable": {"session_id": sessionid}, "callbacks": [lf_handler]})
        r = ''
        for c in response.content:
            r+=c
            time.sleep(0.001)
            yield r   
    else:
        translation = txt.translatef()
        if not truth:
            response = runnable_with_history.invoke({"input": translation}, config={"configurable": {"session_id": sessionid}})##CONFIGURE!
        else:
            response = runnable_with_history.invoke({"input": translation}, config={"configurable": {"session_id": sessionid}, "callbacks": [lf_handler]})
        t = Translation(response.content, txt.original)
        res = t.translatef()
        r = ''
        for c in res:
            r+=c
            time.sleep(0.001)
            yield r
    
    
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

user_lf_host = gr.Textbox(label="LangFuse Host",info="Provide LangFuse host URL, or type 'None' if you do not wish to use LangFuse",value="https://cloud.langfuse.com")

user_lf_pkey = gr.Textbox(label="LangFuse Public Key",info="Provide LangFuse Public key, or type 'None' if you do not wish to use LangFuse",value="pk-*************************", type="password")

user_lf_skey = gr.Textbox(label="LangFuse Secret Key",info="Provide LangFuse Secret key, or type 'None' if you do not wish to use LangFuse",value="sk-*************************", type="password")

user_template = gr.Textbox(label="System Template",info="Customize your assistant with your instructions",value="You are an helpful assistant")

user_session_id = gr.Textbox(label="Session ID",info="This alphanumeric code will link model reply to a specific message history of which the models will be aware when replying. Changing it will result in the loss of memory for your model",value="1")

additional_accordion = gr.Accordion(label="Parameters to be set before you start chatting", open=True)

demo = gr.ChatInterface(fn=reply, additional_inputs=[chat_model, user_api_key, user_temperature, user_max_new_tokens, user_lf_host, user_lf_pkey, user_lf_skey, user_template, user_session_id], additional_inputs_accordion=additional_accordion, title="everything-ai-simplychatting")


if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", share=False)

