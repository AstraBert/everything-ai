from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import time
from langchain_community.llms import HuggingFacePipeline
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os
from pypdf import PdfMerger
from argparse import ArgumentParser


argparse = ArgumentParser()
argparse.add_argument(
    "-m",
    "--model",
    help="HuggingFace Model identifier, such as 'google/flan-t5-base'",
    required=True,
)

argparse.add_argument(
    "-t",
    "--task",
    help="Task for the model: for now supported task are ['text-generation', 'text2text-generation']",
    required=True,
)

args = argparse.parse_args()


mod = args.model
tsk = args.task

mod = mod.replace("\"", "").replace("'", "")
tsk = tsk.replace("\"", "").replace("'", "")

TASK_TO_MODEL = {"text-generation": AutoModelForCausalLM, "text2text-generation": AutoModelForSeq2SeqLM}

if tsk not in TASK_TO_MODEL:
    raise Exception("Unsopported task! Supported task are ['text-generation', 'text2text-generation']")

def merge_pdfs(pdfs: list):
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(f"{pdfs[-1].split('.')[0]}_results.pdf")
    merger.close()
    return f"{pdfs[-1].split('.')[0]}_results.pdf"

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

def convert_none_to_str(l: list):
    newlist = []
    for i in range(len(l)):
        if l[i] is None or type(l[i])==tuple:
            newlist.append("")
        else:
            newlist.append(l[i])
    return tuple(newlist)

def just_chatting(
    task,
    model,
    tokenizer,
    query,
    vectordb,
    chat_history=[]
):
    """
    Implements a chat system using Hugging Face models and a persistent database.

    Args:
        task (str): Task for the pipeline; for now supported task are ['text-generation', 'text2text-generation']
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
    pipe = pipeline(task,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens = 2048,
                    repetition_penalty = float(1.2),
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


try:
    tokenizer = AutoTokenizer.from_pretrained(
        mod,
    )


    model = TASK_TO_MODEL[tsk].from_pretrained(
        mod,
    )
except Exception as e:
    import sys
    print(f"The error {e} occured while handling model and tokenizer loading: please ensure that the model you provided was correct and suitable for the specified task. Be also sure that the HF repository for the loaded model contains all the necessary files.", file=sys.stderr)
    sys.exit(1)


