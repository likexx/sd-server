from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import time, torch

def elapsed_time(pipeline, prompt, nb_pass=10, num_inference_steps=20):
    # warmup
    images = pipeline(prompt, num_inference_steps=10).images
    start = time.time()
    for _ in range(nb_pass):
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np")
    end = time.time()
    return (end - start) / nb_pass

# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = StableDiffusionPipeline.from_pretrained(model_id)

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                            # torch_dtype=torch.float16, 
                                                            use_safetensors=True, 
                                                            variant="fp16", 
                                                            safety_checker = None, 
                                                            requires_safety_checker = False
                                                            )


prompt = "sailing ship in storm by Rembrandt"

result = pipeline(prompt, num_inference_steps=10, output_type="np").images()
