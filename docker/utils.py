# f(x)s that now are useful for all the tasks
from langdetect import detect
from deep_translator import GoogleTranslator
from pypdf import PdfMerger
from qdrant_client import models
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

def remove_items(test_list, item): 
    res = [i for i in test_list if i != item] 
    return res 

def merge_pdfs(pdfs: list):
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(f"{pdfs[-1].split('.')[0]}_results.pdf")
    merger.close()
    return f"{pdfs[-1].split('.')[0]}_results.pdf"

class NeuralSearcher:
    def __init__(self, collection_name, client, model):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = model
        # initialize Qdrant client
        self.qdrant_client = client
    def search(self, text: str):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=1,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads

class PDFdatabase:
    def __init__(self, pdfs, encoder, client):
        self.finalpdf = merge_pdfs(pdfs)
        self.collection_name = os.path.basename(self.finalpdf).split(".")[0].lower()
        self.encoder = encoder
        self.client = client
    def preprocess(self):
        loader = PyPDFLoader(self.finalpdf)
        documents = loader.load()
        ### Split the documents into smaller chunks for processing
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.pages = text_splitter.split_documents(documents)
    def collect_data(self):
        self.documents = []
        for text in self.pages:
            contents = text.page_content.split("\n")
            contents = remove_items(contents, "")
            for content in contents:
                self.documents.append({"text": content, "source": text.metadata["source"], "page": str(text.metadata["page"])})
    def qdrant_collection_and_upload(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        )
        self.client.upload_points(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=idx, vector=self.encoder.encode(doc["text"]).tolist(), payload=doc
                )
                for idx, doc in enumerate(self.documents)
            ],
        )

class Translation:
    def __init__(self, text, destination):
        self.text = text
        self.destination = destination
        try:
            self.original = detect(self.text)
        except Exception as e:
            self.original = "auto"
    def translatef(self):
        translator = GoogleTranslator(source=self.original, target=self.destination)
        translation = translator.translate(self.text)
        return translation

