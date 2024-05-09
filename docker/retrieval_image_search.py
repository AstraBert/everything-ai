from transformers import AutoImageProcessor, AutoModel
from utils import ImageDB
from PIL import Image
from qdrant_client import QdrantClient
import gradio as gr
from argparse import ArgumentParser
import torch

argparse = ArgumentParser()
argparse.add_argument(
    "-m",
    "--model",
    help="HuggingFace Model identifier, such as 'google/flan-t5-base'",
    required=True,
)

argparse.add_argument(
    "-id",
    "--image_dimension",
    help="Dimension of the image (e.g. 512, 758, 384...)",
    required=False,
    default=512,
    type=int
)

argparse.add_argument(
    "-d",
    "--directory",
    help="Directory where all your pdfs of interest are stored",
    required=False,
    default="No directory"
)


args = argparse.parse_args()


mod = args.model
dirs = args.directory
imd = args.image_dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(mod)
model = AutoModel.from_pretrained(mod).to(device)

client = QdrantClient(host="host.docker.internal", port=6333)
imdb = ImageDB(dirs, processor, model, client, imd)
print(imdb.collection_name)
imdb.create_dataset()
imdb.to_collection()


def see_images(dataset, results):
    images = []
    for i in range(len(results)):
        img = dataset[results[0].id]['image']
        images.append(img)
    return images

def process_img(image):
    global imdb
    results = imdb.searchDB(Image.fromarray(image))
    images = see_images(imdb.dataset, results)
    return images


iface = gr.Interface(
    title="everything-ai-retrievalimg",
    fn=process_img,
    inputs=gr.Image(label="Input Image"),
    outputs=gr.Gallery(label="Matching Images"),  
)

iface.launch(server_name="0.0.0.0", share=False)