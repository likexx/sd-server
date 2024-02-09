from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("mps")

NEGATIVE_PROMPT="(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, ((bad hands)), text, error, missing fingers, extra digit, fewer digits,head out of frame, cropped, letterboxed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, letterbox, blurry, monochrome, fused clothes, nail polish, boring, extra legs, fused legs, missing legs, missing arms, extra arms, fused arms, missing limbs, mutated limbs, dead eyes, empty eyes, 2girls, multiple girls, 1boy, 2boys, multiple boys, multiple views, jpeg artifacts, text, signature, watermark, artist name, logo, low res background, low quality background, missing background, white background,deformed"

pipeline.load_lora_weights("/Users/likezhang/projects/models/likezhang.safetensors")
prompt = "likezhang is looking at you. likezhang is 35 years old. {likezhang},front face, portrait, close up, best quality, masterpiece, realistic,masterpiece,vivid,realistic,photorealistic"
images = pipeline(
            prompt = prompt,
            negative_prompt = NEGATIVE_PROMPT,
            num_images_per_prompt=4,
            num_inference_steps=100                
            ).images

i = 0
for img in images:
    img.save("test_{}.png".format(i))
    i+=1

