from aiohttp import web
import torch
import base64, os, json, uuid, time, threading, subprocess
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



WATERMARK_FONT = ImageFont.truetype("Arial.ttf", 30)

SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "123456")

NUM_IMAGES = 4
IMG_WIDTH = 480
IMG_HEIGHT = 480
INFER_STEPS = 50

modelMap = {
    "cartoon": { "model": "/mnt/disk/model/model_anything/AnythingV5Ink_ink.safetensors" },
    "cartoon-adult": { "model": "/mnt/disk/model/model_anything/AnythingV5Ink_ink.safetensors" },
    "real": {"model": "runwayml/stable-diffusion-v1-5"},
    "real-adult": {"model": "runwayml/stable-diffusion-v1-5"},
    "cartoon-everything-adult": { "model": "/mnt/disk/model/anything_everything/anythingAndEverything.safetensors" }
}

PROMPT_SUGGESTION = [
    'best quality',
    'realistic',
    'masterpiece',
    'vivid',
    'vibrant colors',
    'photorealistic',
]

NEGATIVE_PROMPT="(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, ((bad hands)), text, error, missing fingers, extra digit, fewer digits,head out of frame, cropped, letterboxed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, letterbox, blurry, monochrome, fused clothes, nail polish, boring, extra legs, fused legs, missing legs, missing arms, extra arms, fused arms, missing limbs, mutated limbs, dead eyes, empty eyes, 2girls, multiple girls, 1boy, 2boys, multiple boys, multiple views, jpeg artifacts, text, signature, watermark, artist name, logo, low res background, low quality background, missing background, white background,deformed"

def createPipeline(style):
    if not style in modelMap:
        print("invalid style: " + style)
        print("fallback to use anything (cartoon)")
        style = "cartoon"

    requireSafetyChecker = True    
    if style.endswith('-adult'):
        # edit StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
        safety_checker.StableDiffusionSafetyChecker.forward = sc
        requireSafetyChecker = False

    model = modelMap[style]["model"]
    lora = modelMap[style].get("lora", None)
    if model.endswith('.safetensors') or model.endswith('.ckpt'):
        if not requireSafetyChecker:
            pipeline = StableDiffusionPipeline.from_single_file(model, safety_checker = None, requires_safety_checker = False)
        else:
            pipeline = StableDiffusionPipeline.from_single_file(model)
        components = pipeline.components
        img2imgPipeline = StableDiffusionImg2ImgPipeline(**components)     
    elif model=='sdxl':
        if not requireSafetyChecker:
            pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True, 
                                                            variant="fp16", 
                                                            safety_checker = None, 
                                                            requires_safety_checker = False
                                                            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True, 
                                                            variant="fp16"
                                                            )

        components = pipeline.components
        img2imgPipeline = StableDiffusionXLImg2ImgPipeline(**components)
    else:
        if not requireSafetyChecker:
            pipeline = StableDiffusionPipeline.from_pretrained(model, 
                                                                revision="fp16", 
                                                                torch_dtype=torch.float16, 
                                                                safety_checker = None, 
                                                                requires_safety_checker = False)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(model, 
                                                                revision="fp16", 
                                                                torch_dtype=torch.float16)

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

    if pipeline:
        pipeline.to("cuda")

    if img2imgPipeline:
        img2imgPipeline.to("cuda")

    return pipeline, img2imgPipeline

def generate(
            prompt, 
            negPrompt = NEGATIVE_PROMPT, 
            image=None, 
            steps=INFER_STEPS,
            numImages=NUM_IMAGES,
            style = "cartoon"
            ):

    result = []
    
    enhancedPrompt = prompt
    for w in PROMPT_SUGGESTION:
        if enhancedPrompt.find(w) < 0:
            enhancedPrompt += "," + w
    print(enhancedPrompt)

    txt2imgPipeline, img2imgPipeline = createPipeline(style)
    
    if not image or not img2imgPipeline:
        print("generate with txt2img")
        images = txt2imgPipeline(enhancedPrompt,
                            negative_prompt=NEGATIVE_PROMPT,
                            num_images_per_prompt=numImages,
                            num_inference_steps=steps,
                            height=IMG_HEIGHT,
                            width=IMG_WIDTH).images
    else:
        print("generate with img2img")
        init_image = imgUtil.base64_to_rgb_image(image)
        init_image = init_image.resize((IMG_WIDTH, IMG_HEIGHT))
        images = img2imgPipeline(enhancedPrompt,
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

def worker():
    try:
        job = consumer.getNextAvailableJob()
        if not job:
            print("no job found")
        if job:
            print(job)
            jobId = job['job_id']
            try:
                consumer.updateJobStatus(jobId, "generating")
                config = job['job_config']
                # config = json.loads(jobConfigStr)
                imagefile = config.get('image_file', None)
                bucketName = config.get('bucket', None)
                imageData = None
                if imagefile and bucketName:
                    imageData = bucket.read_file_from_bucket(bucket_name=bucketName, blob_name=imagefile)
                prompt = config.get('prompt', '')
                negPrompt = config.get('negative_prompt', '')
                steps = config.get('steps', 50)
                numImages = config.get('num_images', 8)
                size = config.get('size', 360)
                style = config.get('style', 'cartoon')

                if style == 'realistic':
                    style = 'real'
                elif style == 'realistic-adult':
                    style = 'real-adult'
                
                images = generate(prompt=prompt, 
                                  negPrompt=negPrompt, 
                                  image=imageData, 
                                  steps=steps, 
                                  numImages=numImages, 
                                  style=style)
                result = []
                for image in images:
                    d = image['base64_str']
                    aigcBucketName = bucket.aigc_img_bucket_name
                    aigcFilename = jobId + '-' + str(uuid.uuid4()).replace('-', '')
                    bucket.upload_to_bucket(d, aigcBucketName, aigcFilename)
                    result.append({
                        'bucket': aigcBucketName,
                        'image_file': aigcFilename
                    })
                updateResult = consumer.updateJobResult(jobId=jobId, result=json.dumps(result))
                print("job result saved", updateResult, jobId, result)
                print("****************job done:", jobId)
            except Exception as err:
                print(err)
                consumer.updateJobStatus(jobId, "failed")

    except Exception as e:
        print(e)


if __name__ == '__main__':
    worker()
