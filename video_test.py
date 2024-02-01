import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from diffusers.pipelines.stable_diffusion import safety_checker

def remove_nsfw_check(self, clip_input, images) :
    return images, [False for i in images]


safety_checker.StableDiffusionSafetyChecker.forward = remove_nsfw_check

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", variant="fp16",
    use_safetensors=True, safety_checker = None, requires_safety_checker = False
)
#pipe.enable_model_cpu_offload()
pipe.to('cpu') # Force to GPU

# Load the conditioning image
image = load_image("./output/3704.png")
image = image.resize((512, 288))

generator = torch.manual_seed(32)

# Perform GPU memory cleanup
# gc.collect()
# torch.cuda.empty_cache()

decode_chunk_size = 2
num_frames = 10
motion_bucket_id=180
noise_aug_strength = 0.3

frames = pipe(image, 
              decode_chunk_size=decode_chunk_size, 
              generator=generator,
              motion_bucket_id=180,
              noise_aug_strength=0.1).frames[0]

print("generating frames done. export")

export_to_video(frames, "./output/generated-6.mp4", fps=7)

print("done")