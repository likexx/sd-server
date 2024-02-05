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

for i in range(1, 9):
    img = Image.open('../hed/{}.png'.format(i))
    print(img.size)
    img = img.resize((256, 256))
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
edges = edges + edges[::-1]
# edges = edges[:12]



# model_id = "runwayml/stable-diffusion-v1-5"
model_name = sys.argv[1]
print("using model: " +model_name)
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
latents = torch.randn((1, 4, 32, 32), device="cuda", dtype=torch.float16).repeat(len(edges), 1, 1, 1)

prompt = '''
1 girl, an ancient chinese girl is crunching on the bed and raising her ass high. view from aside, long shot. The character has beautiful face, round eyes, and long hair. She is screaming. She is wearing white classic chinese dress, half naked, large breast, legs naked and vagina exposed, hands on the bed, raising her ass high, kneeing on the bed. The character has slim waist, beautiful legs,long black hair. Her legs are slightly open. She is being fucked from behind. master piece, detailed, vivid, colorful, masterpiece, high quality
'''
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
weighted_prompt = compel([prompt] * len(edges))
result = pipe(prompt_embeds=weighted_prompt, pooled_prompt_embeds = None, image=edges, latents=latents, width=256, height=256, num_inference_steps=200).images
imageio.mimsave("video-1.mp4", result, fps=8)

