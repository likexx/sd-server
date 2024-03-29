from huggingface_hub import hf_hub_download
from PIL import Image
import imageio
import torch
import numpy as np
from diffusers import TextToVideoZeroPipeline, TextToVideoZeroSDXLPipeline
from diffusers.pipelines.stable_diffusion import safety_checker
import cv2
from compel import Compel, ReturnedEmbeddingsType
import sys

def remove_nsfw_check(self, clip_input, images) :
    return images, [False for i in images]


safety_checker.StableDiffusionSafetyChecker.forward = remove_nsfw_check

model_name = sys.argv[1]
print("using model: " +model_name)
# model_id = "/home/likezhang/models/{}.safetensors".format(model_name)
model_id = "stablediffusionapi/anything-v5"
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = TextToVideoZeroPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    safety_checker=None, 
    use_safetensors=True).to("cuda")

seed = 0
video_length = 24  #24 ÷ 4fps = 6 seconds
chunk_size = 8
prompt = ['''
one girl is crunching on the bed and raising her ass upwards, being fucked, kneeing on bed, hands on bed, face downward, view from side, long shot, master piece, high quality, vivid color, masterpiece, details
''',
'''
one girl is crunching on the bed and raising her ass downwards, yelling, being fucked, kneeing on bed, hands on bed, face upward, view from side, long shot, master piece, high quality, vivid color, masterpiece, details
'''
]
# compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
# weighted_prompt = compel([prompt])
# compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , 
#                 text_encoder=[pipe.text_encoder, pipe.text_encoder_2], 
#                 returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
#                 requires_pooled=[False, True])

# weighted_prompt, pooled = compel([prompt])

result = []
chunk_ids = np.arange(0, video_length, chunk_size - 1)
generator = torch.Generator(device="cuda")
for i in range(len(chunk_ids)):
    print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
    ch_start = chunk_ids[i]
    ch_end = video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
    # Attach the first frame for Cross Frame Attention
    frame_ids = [0] + list(range(ch_start, ch_end))
    # Fix the seed for the temporal consistency
    generator.manual_seed(seed)
    promp = prompt[i%2]
    output = pipe(
                  prompt = promp,
                  video_length=len(frame_ids), 
                  generator=generator, width=256, height=256, 
                  motion_field_strength_x = 0,
                  motion_field_strength_y = 0,
                  num_inference_steps=100, 
                  frame_ids=frame_ids)
    result.append(output.images[1:])

# Concatenate chunks and save
result = np.concatenate(result)
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video.mp4", result, fps=4)


