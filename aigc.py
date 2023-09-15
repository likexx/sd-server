import torch
import base64, os, json, uuid, time
from io import BytesIO
import argparse
from PIL import Image, ImageDraw, ImageFont
import img_util as imgUtil

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker
import consumer
import gcloud_bucket as bucket

def sc(self, clip_input, images) :
    return images, [False for i in images]

# edit StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc

pipeline_lock = threading.Lock()

WATERMARK_FONT = ImageFont.truetype("Arial.ttf", 30)

SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "123456")
NUM_OF_IMAGES = 4
HEIGHT=480
WIDTH=480
STEPS = 50

modelMap = {
    "cartoon": { "model": "/mnt/disk/model/model_anything/AnythingV5Ink_ink.safetensors" },
    "cartoon-adult": { "model": "/mnt/disk/model/model_anything/AnythingV5Ink_ink.safetensors" },
    "real": {"model": "runwayml/stable-diffusion-v1-5"},
    "real-adult": {"model": "runwayml/stable-diffusion-v1-5"}
}

NEGATIVE_PROMPT="(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, ((bad hands)), text, error, missing fingers, extra digit, fewer digits,head out of frame, cropped, letterboxed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, letterbox, blurry, monochrome, fused clothes, nail polish, boring, extra legs, fused legs, missing legs, missing arms, extra arms, fused arms, missing limbs, mutated limbs, dead eyes, empty eyes, 2girls, multiple girls, 1boy, 2boys, multiple boys, multiple views, jpeg artifacts, text, signature, watermark, artist name, logo, low res background, low quality background, missing background, white background,deformed"

NEW_JOB_FILENAME = "./input/new_job_start"
JOB_CONFIG_FILENAME = "./input/job.json"
IMAGE_DATA_FILENAME = "./input/image.data"
OUTPUT_PATH = "./output"

def createPipeline(style):
    if not style in modelMap:
        print("invalid style: " + style)
        print("fallback to use anything (cartoon)")
        style = "cartoon"

    model = modelMap[style]["model"]
    lora = modelMap[style].get("lora", None)
    if model.endswith('.safetensors') or model.endswith('.ckpt'):
        pipeline = StableDiffusionPipeline.from_single_file(model, safety_checker = None, requires_safety_checker = False)
        components = pipeline.components
        img2imgPipeline = StableDiffusionImg2ImgPipeline(**components)     
    elif model=='sdxl':
        pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                        torch_dtype=torch.float16, 
                                                        use_safetensors=True, 
                                                        variant="fp16", 
                                                        safety_checker = None, 
                                                        requires_safety_checker = False
                                                        )
        components = pipeline.components
        img2imgPipeline = StableDiffusionXLImg2ImgPipeline(**components)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model, 
                                                            revision="fp16", 
                                                            torch_dtype=torch.float16, 
                                                            safety_checker = None, 
                                                            requires_safety_checker = False)
        components = pipeline.components
        img2imgPipeline = StableDiffusionImg2ImgPipeline(**components)     
        # img2imgPipeline = StableDiffusionImg2ImgPipeline(model,
        #                                                  revision="fp16",
        #                                                  torch_dtype=torch.float16, 
        #                                                  safety_checker = None,
        #                                                  requires_safety_checker = False)        
    if lora:
        if lora.endswith('.safetensors'):
            print("load lora weights")
            pipeline.load_lora_weights(".", weight_name=lora)
        else:
            pipeline.unet.load_attn_procs(lora)

    pipeline.to("cuda")
    if img2imgPipeline:
        img2imgPipeline.to("cuda")

    return pipeline, img2imgPipeline

def generate(
            txt2imgPipeline,
            img2imgPipeline,
            prompt, 
            negPrompt = NEGATIVE_PROMPT, 
            image=None, 
            steps=50,
            numImages=NUM_OF_IMAGES,
            style = "cartoon"
            ):
    result = []
    if not image or not img2imgPipeline:
        images = txt2imgPipeline(prompt,
                            negative_prompt=NEGATIVE_PROMPT,
                            num_images_per_prompt=numImages,
                            num_inference_steps=steps,
                            height=HEIGHT,
                            width=WIDTH).images
    else:
        init_image = imgUtil.base64_to_rgb_image(image)
        init_image = init_image.resize((WIDTH, HEIGHT))
        images = img2imgPipeline(prompt,
                                image=init_image,
                                negative_prompt=NEGATIVE_PROMPT,
                                num_images_per_prompt=numImages,
                                num_inference_steps=steps,
                                ).images
    for img in images:
        buffered = BytesIO()
        # finalImage = imgUtil.add_watermark(img, "Created by KK Studio")
        img.save(buffered, format="JPEG")
        # finalImage.save(buffered, format="JPEG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        result.append({'base64_str': base64_str})
    return result

def createImages(prompt, negPrompt, imageData, steps, numImages, style):                    
    images = generate(prompt=prompt, negPrompt=negPrompt, image=imageData, steps=steps, numImages=numImages, style = style)
    i = 0
    for image in images:
        d = image['base64_str']
        with open('./output/output{}.data'.format(i), 'w') as f:
            f.write(d)
    with open('./output/complete', 'w') as f:
        f.write("done")

def initializePipeline(config):
    imageData = None
    jobData = None
    imageFilepath = os.path.join(config.input, IMAGE_DATA_FILENAME)
    jobFilepath = os.path.join(config.input, JOB_CONFIG_FILENAME)

    with open(imageFilepath, 'r') as f:
        imageData = f.read()
    
    with open(jobFilepath, 'r') as f:
        jobData = json.load(f)
    
    style = jobData["style"]
    p1, p2 =  createPipeline(style)
    return jobData, imageData, p1, p2

if __name__ == '__main__':
    count = 0
    while not os.path.exists(NEW_JOB_FILENAME):
        if count >= 300:
            print("new job not found")
            count = 0
        time.sleep(2)
        count += 1
        continue
    
    job, pipeline, imageData, img2imgPipeline = initializePipeline()
    createImages(pipeline, img2imgPipeline, job['prompt'], job['negprompt'], imageData, job['steps'], job['images'])
