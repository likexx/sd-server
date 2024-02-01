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
# pipe = pipe.to('cuda') # Force to GPU
pipe.enable_model_cpu_offload()

# Load the conditioning image

generator = torch.manual_seed(32)

# Perform GPU memory cleanup
# gc.collect()
# torch.cuda.empty_cache()

decode_chunk_size = 2
num_frames = 10
motion_bucket_id=180
noise_aug_strength = 0.3

images = [
    '01','02','03','04','05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17'
]

for img_name in images:
    image_path = '/home/likezhang/temp/asset/{}.png'.format(img_name)
    image = load_image(image_path)
    image = image.resize((512, 288))

    frames = pipe(image, 
                decode_chunk_size=decode_chunk_size, 
                generator=generator,
                motion_bucket_id=180,
                noise_aug_strength=0.1).frames[0]
    output_path = "./output/generated-{}.mp4".format(img_name)

    print("generating frames done. export " + output_path)

    export_to_video(frames, output_path, fps=7)

    print("done")