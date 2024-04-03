# -*- coding: utf-8 -*-

"""# gemma-2b AS A DATA SCIENCE TEACHER

## 100% LOCAL, WITHOUT FINETUNING, WITH _YOUR OWN DATA_

> _Information is the oil of the 21st century, and analytics is the combustion engine – Peter Sondergaard (Senior Vice President and the Global Head of Research at Gartner Inc)_

In a world where data are becoming more important with each day passing, data science is a fundamental discipline to master in order to understand and solve the upcoming challenges of the Big Data World.

Unfortunately, data science is generally available to University-level students only, making it difficult for other people to access its concepts. This obstacle can be removed with the help of Large Language Models, such as _gemma-2b_.

In this notebook, we'll make our way through the jungle of data science thanks to _gemma-2b_, a simple pdf file titled **"What is data science?"**, ChromaDB vectorstores and Langchain, all elengatly written in python.

The final goal is to implement a simple, yet powerful, pipeline to generate a 100% local and fully-customizable LLM-based assistant that works with the user's data.

Let's dive in!🛫



# Build the environment

First of all, we want everything set up the right way to work properly. To do so, we need to:

1. Upload the pdf file in our workspace (we can simply create a dataset in Kaggle containing the pdf and add it as `input` to the notebook): in the following notebook example, we will name it "/kaggle/input/what-is-datascience-docs/WhatisDataScienceFinalMay162018.pdf".
2. Install necessary dependencies
3. Upload _gemma-2b_ model as Kaggle input
4. Define useful functions to make our LLM-based data science assistant work
"""

# IMPORT gemma-2b MODEL FROM KAGGLE

## To import the model, we'll be uploading the model directly from Kaggle input

from transformers import AutoTokenizer, AutoModelForCausalLM
model_checkpoint = "/kaggle/input/gemma/transformers/2b/1"

hf_token = "YOUR_TOKEN"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, token=hf_token)

# DEFINE USEFUL FUNCTIONS

## To chat, we'll need to create a vectorized database from our pdf and then build
## a retrieval Q&A chain

import time
from langchain_community.llms import HuggingFacePipeline
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

def create_a_persistent_db(pdfpath, dbpath, cachepath) -> None:
    """
    Creates a persistent database from a PDF file.

    Args:
        pdfpath (str): The path to the PDF file.
        dbpath (str): The path to the storage folder for the persistent LocalDB.
        cachepath (str): The path to the storage folder for the embeddings cache.
    """
    print("Started the operation...")
    a = time.time()
    loader = PyPDFLoader(pdfpath)
    documents = loader.load()

    ### Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    ### Use HuggingFace embeddings for transforming text into numerical vectors
    ### This operation can take a while the first time but, once you created your local database with
    ### cached embeddings, it should be a matter of seconds to load them!
    embeddings = HuggingFaceEmbeddings()
    store = LocalFileStore(
        os.path.join(
            cachepath, os.path.basename(pdfpath).split(".")[0] + "_cache"
        )
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=store,
        namespace=os.path.basename(pdfpath).split(".")[0],
    )

    b = time.time()
    print(
        f"Embeddings successfully created and stored at {os.path.join(cachepath, os.path.basename(pdfpath).split('.')[0]+'_cache')} under namespace: {os.path.basename(pdfpath).split('.')[0]}"
    )
    print(f"To load and embed, it took: {b - a}")

    persist_directory = os.path.join(
        dbpath, os.path.basename(pdfpath).split(".")[0] + "_localDB"
    )
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=cached_embeddings,
        persist_directory=persist_directory,
    )
    c = time.time()
    print(
        f"Persistent database successfully created and stored at {os.path.join(dbpath, os.path.basename(pdfpath).split('.')[0] + '_localDB')}"
    )
    print(f"To create a persistent database, it took: {c - b}")
    return vectordb

def just_chatting(
    model,
    tokenizer,
    query,
    vectordb,
    chat_history=[]
):
    """
    Implements a chat system using Hugging Face models and a persistent database.

    Args:
        model (AutoModelForCausalLM): Hugging Face model, already loaded and prepared.
        tokenizer (AutoTokenizer): Hugging Face tokenizer, already loaded and prepared.
        model_task (str): Task for the Hugging Face model.
        persistent_db_dir (str): Directory for the persistent database.
        embeddings_cache (str): Path to cache Hugging Face embeddings.
        pdfpath (str): Path to the PDF file.
        query (str): Question by the user
        vectordb (ChromaDB): vectorstorer variable for retrieval.
        chat_history (list): A list with previous questions and answers, serves as context; by default it is empty (it may make the model allucinate)
    """
    ### Create a text-generation pipeline and connect it to a ConversationalRetrievalChain
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens = 2048,
                    repetition_penalty = float(10),
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    llm_chain = ConversationalRetrievalChain.from_llm(
        llm=local_llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=False,
    )
    rst = llm_chain({"question": query, "chat_history": chat_history})
    return rst

"""# Chat with the model

To chat with the model, we first have to build our local, persistent, database, and also compute embeddings: after that, we'll be able to chat with the model without problems!🚀
"""

# CREATE PERSISTENT DB

filepath = "/kaggle/input/what-is-datascience-docs/WhatisDataScienceFinalMay162018.pdf"
dbpath = "/kaggle/working/"
cachepath = "/kaggle/working/"
vectordb = create_a_persistent_db(filepath, dbpath, cachepath)

# CHAT WITH MODEL

chat_history = []
query = "Define datascience"
res = just_chatting(model, tokenizer, query, vectordb, chat_history=chat_history)
chat_history.append([query, res["answer"].replace("\n"," ")])

print(" ".join[res["answer"]])

"""# Implement a simple chat GUI (local only)

Want to interact more directly with your model, without going through that pythonic stuff? Let's implement a very simple and rudimental chat GUI, based on builtin package `tkinter`, to achieve this goal!🤯
"""

import tkinter as tk
from tkinter import scrolledtext

class ChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("DataScienceAI")

        self.chat_history = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=15)
        self.chat_history.pack(padx=10, pady=10)

        self.user_input = tk.Entry(master, width=40)
        self.user_input.pack(padx=10, pady=10)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(pady=10)

        # Set up initial conversation
        self.display_message("DataScienceAI: Hello! How can I help you today?")

    def send_message(self):
        user_message = self.user_input.get()
        self.display_message(f"You: {user_message}")
        # Replace the next line with your chatbot logic to get a response
        chatbot_response = f"DataScienceAI: {just_chatting(model, tokenizer, user_message, vectordb)["answer"].replace("\n"," ")}"
        self.display_message(chatbot_response)
        self.user_input.delete(0, tk.END)  # Clear the input field

    def display_message(self, message):
        self.chat_history.insert(tk.END, message + '\n')
        self.chat_history.see(tk.END)  # Scroll to the bottom

if __name__ == "__main__":
    root = tk.Tk()
    chat_gui = ChatGUI(root)
    root.mainloop()

"""# Conclusions

This is it!

We built a simple assistant, fully customizable in terms of both the LLM employed (you can switch to _gemma-7b_ or to your favorite LLM) and the data you can make it work with (in this case is data sciences, but you can make it work also on a pdf about pallas' cats, if you want!)🐈.

Another important thing to note is that all of this is completely local, there is no need for hosted APIs, pay-as-you-go services or other things like that: everything is free to use, on your Desktop!

There are two main disadvantages in this approach:

1. Performance-critical tasks, such as loading the model and making prediction, are heavily resource-dependent: to load big models (>1~2 GB) and to make them generate text, it is useful to have more than 16GB RAM and more than 4 CPU cores.
2. Small (and old) models, such as _openai-community/gpt2_, can easily allucinate while generating text. This is generally prompt-dependent (meaning that they tend to produce trashy results on certain prompts more frequently than on other ones) and the issue almost totally resolves when employing large LLMs (_gemma-7b_ or _llama-7b_ would not-so-easily allucinate, for instance).

### TLDR😵:

**Pros**:
- Simple and customizable
- Use virtually any LLM you want
- Use your own data
- 100% local, 100% free, no payments or APIs

**Cons**:
- Performance might be resource-dependent for large LLMs (if you have >16GB RAM and >4 cores it shouldn't be a great problem)
- Small LLMs can still allucinate

# References

- Paul Mooney, Ashley Chow. (2024). Google – AI Assistants for Data Tasks with Gemma. Kaggle. https://kaggle.com/competitions/data-assistants-with-gemma
- Brodie, Michael. (2019). What Is Data Science?. 10.1007/978-3-030-11821-1_8.
"""