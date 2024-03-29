from huggingface_hub import hf_hub_download
from PIL import Image
import imageio
import torch
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from diffusers.pipelines.stable_diffusion import safety_checker
import cv2
from compel import Compel, ReturnedEmbeddingsType
import sys
import argparse

def remove_nsfw_check(self, clip_input, images) :
    return images, [False for i in images]


safety_checker.StableDiffusionSafetyChecker.forward = remove_nsfw_check

# filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
# repo_id = "PAIR/Text2Video-Zero"
# video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
# print(video_path)

# reader = imageio.get_reader('./input/tt3.mp4', "ffmpeg")
# frame_count = 30*10
# pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

edges = []

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--fps', type=int, default=8)
parser.add_argument('--model', type=str, default='anything')
parser.add_argument('--controlnet_scale', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--steps', type=int, default=50)

args = parser.parse_args()

size = args.size
FPS = args.fps
model_name = args.model
controlnet_scale = args.controlnet_scale
seed = args.seed
steps = args.steps
print("using model: {}, controlnet scale: {}, fps: {}, seed: {}, size: {}, steps: {}".format(model_name, controlnet_scale, FPS, seed, size, steps))

# i = 1
# j = 0
# for img in pose_images:
#     print(i)
#     img = img.resize((512, 512))
#     # img.save("./input/pose_{}.png".format(j), 'PNG')
#     data = np.array(img)
#     edge = cv2.Canny(data, 100, 300)
#     edge = edge[:, :, None]
#     edge = np.concatenate([edge, edge, edge], axis=2)
#     edge = Image.fromarray(edge)
#     edge.save("./output/tt3_pose_{}.png".format(i), 'PNG')
#     # edges.append(edge)
#     i+=1    

for i in range(10, 22):
    img = Image.open('../hed/hed_{}.png'.format(i))
    print(img.size)
    img = img.resize((size, size))
    edges.append(img)
    # img.save("./input/pose_{}.png".format(j), 'PNG')
    # data = np.array(img)
    # edge = cv2.Canny(data, 50, 50)
    # edge = edge[:, :, None]
    # edge = np.concatenate([edge, edge, edge], axis=2)
    # edge = Image.fromarray(edge)
    # edge.save("./input/pose_{}.png".format(j), 'PNG')
    # edges.append(edge)
    # i+=1
    # j+=1
    # if j > 10:
    #     break
# edges = edges + edges[::-1]
# edges = edges[:12]



# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/home/likezhang/models/{}.safetensors".format(model_name)
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_single_file(
    model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, use_safetensors=True
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
# 32 for 256x256. should be 64 for size 512x512
latents = torch.randn((1, 4, 32 * (size//256), 32*(size//256)), device="cuda", dtype=torch.float16).repeat(len(edges), 1, 1, 1)

# prompt = '''
# 1 girl, an ancient chinese girl in Song dynasty is crunching on the bed and raising her ass high. view from aside, long shot. The character has beautiful face, round eyes, and long hair. She is screaming. She is wearing white classic chinese dress, half naked, large breast, legs naked and vagina exposed, hands on the bed, raising her ass high, kneeing on the bed. The character has slim waist, beautiful legs,long black hair. Her legs are slightly open. She is being fucked from behind.beautiful face, face details, master piece, detailed, vivid, colorful, masterpiece, high quality
# '''
prompt = '''
2 men, 2 male super saiyan warriors are fighting, battle scene, face details, high quality, masterpiece, vivid color, colorful, realistic, photo quality
'''
neg_prompt = '''
(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, ((bad hands)), text, error, missing fingers, extra digit, fewer digits,head out of frame, cropped, letterboxed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, letterbox, blurry, monochrome, fused clothes, nail polish, boring, extra legs, fused legs, missing legs, missing arms, extra arms, fused arms, missing limbs, mutated limbs, dead eyes, empty eyes, 2girls, multiple girls, 1boy, 2boys, multiple boys, multiple views, jpeg artifacts, text, signature, watermark, artist name, logo, low res background, low quality background, missing background, white background,deformed
'''
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
weighted_prompt = compel([prompt] * len(edges))
generator = torch.Generator('cuda').manual_seed(seed)

# negative_prompt_embeds

result = pipe(prompt_embeds=weighted_prompt, pooled_prompt_embeds = None, 
              negative_prompt=[neg_prompt]*len(edges),
              image=edges, latents=latents, width=size, height=size, num_inference_steps=steps,
              generator = generator,
              controlnet_conditioning_scale = controlnet_scale).images
imageio.mimsave("video-1.mp4", result + result[::-1], fps=FPS)

