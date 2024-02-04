from huggingface_hub import hf_hub_download
from PIL import Image
import imageio
import torch
import numpy as np
from diffusers import TextToVideoZeroPipeline, TextToVideoSDPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from diffusers.pipelines.stable_diffusion import safety_checker
import cv2
from compel import Compel, ReturnedEmbeddingsType
import sys
from diffusers.utils import load_image, export_to_video

def remove_nsfw_check(self, clip_input, images) :
    return images, [False for i in images]


safety_checker.StableDiffusionSafetyChecker.forward = remove_nsfw_check

model_name = sys.argv[1]
print("using model: " +model_name)
model_id = "/home/likezhang/models/{}.safetensors".format(model_name)

pipe = TextToVideoSDPipeline.from_single_file(
    model_id, 
    torch_dtype=torch.float16, 
    safety_checker=None, 
    use_safetensors=True).to("cuda")

seed = 0
video_length = 24  #24 ÷ 4fps = 6 seconds
chunk_size = 8
prompt = '''
1 girl, single frame. one girl is crunching on the bed and raising her ass high. view from aside, long shot. The character is all naked, hands on the bed, raising her ass high, kneeing on the bed. The character has slim waist, beautiful legs,long black hair. Her legs are slightly open. She is being fucked from behind. master piece, detailed, vivid, colorful, masterpiece, high quality
'''
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
weighted_prompt = compel([prompt])

generator = torch.Generator(device="cuda")
output = pipe(
    prompt_embeds=weighted_prompt, 
    pooled_prompt_embeds = None, 
    num_frames=8, 
    generator=generator, width=256, height=256, num_inference_steps=100).frames

video_path = export_to_video(output)
path = export_to_video(output, 'video.mp4', fps=7)
