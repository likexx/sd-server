from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("/home/likezhang/output/pytorch_lora_weights.safetensors")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# text_encoder = CLIPTextModel.from_pretrained("path/to/model/checkpoint-100/checkpoint-100/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, dtype=torch.float16,
).to("cuda")

image = pipeline("likezhang is walking on street", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("likezhang.png")