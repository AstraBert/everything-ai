import gradio as gr
import moviepy.editor as mp
from diffusers import DiffusionPipeline
from argparse import ArgumentParser


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

# Load diffusion pipelines
image_pipeline = DiffusionPipeline.from_pretrained(model_checkpoint)
video_pipeline = DiffusionPipeline.from_pretrained(model_checkpoint)

def generate_images(prompt, num_images):
  """Generates images using the image pipeline."""
  images = []
  for _ in range(num_images):
    generated_image = image_pipeline(prompt=prompt).images[0]
    images.append(generated_image)
  return images


def generate_videos(images):
    """Generates videos from a list of images using the video pipeline."""
    videos = []
    for image in images:
        # Wrap the image in a list as expected by the pipeline
        generated_video = video_pipeline(images=[image]).images[0]
        videos.append(generated_video)
    return videos

def combine_videos(video_clips):
  final_clip = mp.concatenate_videoclips(video_clips)
  return final_clip

def generate(prompt):
  images = generate_images(prompt, 2)
  video_clips = generate_videos(images)
  combined_video = combine_videos(video_clips)
  return combined_video

# Gradio interface with improved formatting and video output
interface = gr.Interface(
    fn=generate,
    inputs="text",
    outputs="video",
    title="everything-ai-text2vid",
    description="Enter a prompt to generate a video using diffusion models.",
    css="""
      .output-video {
        width: 100%; /* Adjust width as needed */
        height: 400px; /* Adjust height as desired */
      }
    """,
)

# Launch the interface
interface.launch(server_name="0.0.0.0", share=False)
