from huggingface_hub import hf_hub_download
from PIL import Image
import imageio
import torch
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from diffusers.pipelines.stable_diffusion import safety_checker
import cv2


def remove_nsfw_check(self, clip_input, images) :
    return images, [False for i in images]


safety_checker.StableDiffusionSafetyChecker.forward = remove_nsfw_check

# filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
# repo_id = "PAIR/Text2Video-Zero"
# video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
# print(video_path)

reader = imageio.get_reader('./input/test01.mp4', "ffmpeg")
frame_count = 30*15
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

edges = []

i = 1
j = 0
for img in pose_images:
    if i%30 != 0:
        i+=1
        continue
    print(img.size)
    img = img.resize((512, 512))
    # img.save("./input/pose_{}.png".format(j), 'PNG')
    data = np.array(img)
    edge = cv2.Canny(data, 50, 100)
    edge = edge[:, :, None]
    edge = np.concatenate([edge, edge, edge], axis=2)
    edge = Image.fromarray(edge)
    edge.save("./input/pose_{}.png".format(j), 'PNG')
    edges.append(edge)
    i+=1
    j+=1




# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/home/likezhang/anything.safetensors"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_single_file(
    model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, use_safetensors=True
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(edges), 1, 1, 1)

prompt = "a naked chinese female girl is dancing on a beach"
result = pipe(prompt=[prompt] * len(edges), image=edges, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)

