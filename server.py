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

def initConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, required=True, help='cartoon | realistic')
    parser.add_argument('--lora', type=str, default='', help='lora location')
    parser.add_argument('--token', type=str, default="123456", help='server token')
    parser.add_argument('--steps', type=int, default=50, help='steps')
    parser.add_argument('--width', type=int, default=480, help='image width (same height)')
    # Parse the arguments
    args = parser.parse_args()
    return args


def init():
    global SERVER_TOKEN, STEPS, WIDTH, HEIGHT
    conf = initConfig()
    if conf.token != '':
        SERVER_TOKEN = conf.token

    STEPS = conf.steps
    WIDTH = conf.width
    HEIGHT = WIDTH

def checkAigcResult(style, prompt, negPrompt, steps, images, inputImageData):
    OUTPUT_COMPLETE_FILEPATH = './output/complete'
    waitCount = 0
    while not os.path.exists(OUTPUT_COMPLETE_FILEPATH):
        if waitCount > 300:
            print("not finding completion signal")
            waitCount=0
        waitCount+=1

    jobIdStr = ""
    with open(OUTPUT_COMPLETE_FILEPATH, 'w') as f:
        jobIdStr = f.read()

    images = []
    path = './output'
    for filename in os.listdir(path):
        # Open and read each file
        if filename.endswith('.data'):
            with open(os.path.join(path, filename), 'r') as f:
                data = f.read()
                images.append(data)
    return jobId, images

# def createPipeline(style):
#     if not style in modelMap:
#         print("invalid style: " + style)
#         print("fallback to use anything (cartoon)")
#         style = "cartoon"

#     model = modelMap[style]["model"]
#     lora = modelMap[style].get("lora", None)
#     if model.endswith('.safetensors') or model.endswith('.ckpt'):
#         pipeline = StableDiffusionPipeline.from_single_file(model, safety_checker = None, requires_safety_checker = False)
#         components = pipeline.components
#         img2imgPipeline = StableDiffusionImg2ImgPipeline(**components)     
#     elif model=='sdxl':
#         pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
#                                                         torch_dtype=torch.float16, 
#                                                         use_safetensors=True, 
#                                                         variant="fp16", 
#                                                         safety_checker = None, 
#                                                         requires_safety_checker = False
#                                                         )
#         components = pipeline.components
#         img2imgPipeline = StableDiffusionXLImg2ImgPipeline(**components)
#     else:
#         pipeline = StableDiffusionPipeline.from_pretrained(model, 
#                                                             revision="fp16", 
#                                                             torch_dtype=torch.float16, 
#                                                             safety_checker = None, 
#                                                             requires_safety_checker = False)
#         components = pipeline.components
#         img2imgPipeline = StableDiffusionImg2ImgPipeline(**components)     
#         # img2imgPipeline = StableDiffusionImg2ImgPipeline(model,
#         #                                                  revision="fp16",
#         #                                                  torch_dtype=torch.float16, 
#         #                                                  safety_checker = None,
#         #                                                  requires_safety_checker = False)        
#     if lora:
#         if lora.endswith('.safetensors'):
#             print("load lora weights")
#             pipeline.load_lora_weights(".", weight_name=lora)
#         else:
#             pipeline.unet.load_attn_procs(lora)

#     pipeline.to("cuda")
#     if img2imgPipeline:
#         img2imgPipeline.to("cuda")

#     return pipeline, img2imgPipeline

# def generate(
#             prompt, 
#             negPrompt = NEGATIVE_PROMPT, 
#             image=None, 
#             steps=50,
#             numImages=NUM_OF_IMAGES,
#             style = "cartoon"
#             ):
#     global pipeline_lock

#     result = []
#     with pipeline_lock:
#         txt2imgPipeline, img2imgPipeline = createPipeline(style)
#         if not image or not img2imgPipeline:
#             images = txt2imgPipeline(prompt,
#                                 negative_prompt=NEGATIVE_PROMPT,
#                                 num_images_per_prompt=numImages,
#                                 num_inference_steps=steps,
#                                 height=HEIGHT,
#                                 width=WIDTH).images
#         else:
#             init_image = imgUtil.base64_to_rgb_image(image)
#             init_image = init_image.resize((WIDTH, HEIGHT))
#             images = img2imgPipeline(prompt,
#                                     image=init_image,
#                                     negative_prompt=NEGATIVE_PROMPT,
#                                     num_images_per_prompt=numImages,
#                                     num_inference_steps=steps,
#                                     ).images
#         for img in images:
#             buffered = BytesIO()
#             # finalImage = imgUtil.add_watermark(img, "Created by KK Studio")
#             img.save(buffered, format="JPEG")
#             # finalImage.save(buffered, format="JPEG")
#             base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
#             result.append({'base64_str': base64_str})
#         # print(base64_str)
#     return result

routes = web.RouteTableDef()

@routes.post('/healthcheck')
async def handle(request):
    text = "ok, ai"
    return web.Response(text=text)


# @routes.post('/txt2image')
# async def text_to_image_handle(request):
#     post = await request.json()
#     if post.get('token') != SERVER_TOKEN:
#         return web.json_response({'result': "invalid token"})

#     prompt = post.get("prompt")
#     negPrompt = post.get("neg_prompt")
#     num_images = post.get("number_images")
#     steps = post.get("steps")

#     if num_images:
#         num_images = int(num_images)
#     else:
#         num_images = NUM_OF_IMAGES

#     if steps:
#         steps = int(steps)
#     else:
#         steps = STEPS
    
#     data = generate(prompt=prompt, negPrompt=negPrompt, steps=steps, numImages=num_images)
#     return web.json_response({'result': data})

# @routes.post('/img2image')
# async def img_to_image_handle(request):
#     post = await request.json()
#     if post.get('token') != SERVER_TOKEN:
#         return web.json_response({'result': "invalid token"})

#     prompt = post.get("prompt")
#     negPrompt = post.get("neg_prompt")
#     image = post.get("image_data")
#     num_images = post.get("number_images")
#     steps = post.get("steps")

#     if num_images:
#         num_images = int(num_images)
#     else:
#         num_images = NUM_OF_IMAGES

#     if steps:
#         steps = int(steps)
#     else:
#         steps = STEPS
    
#     data = generate(prompt=prompt, negPrompt=negPrompt, image=image, steps=steps, numImages=num_images)
#     return web.json_response({'result': data})

def aigcJobThread():
    while True:
        try:
            job = consumer.getNextAvailableJob()
            if job:
                print(job)
                jobId = job['job_id']
                try:
                    consumer.updateJobStatus(jobId, "generating")
                    config = job['job_config']
                    # config = json.loads(jobConfigStr)
                    imagefile = config.get('image_file', "")
                    bucketName = config.get('bucket', "")
                    imageData = ""
                    if imagefile and bucketName:
                        imageData = bucket.read_file_from_bucket(bucket_name=bucketName, blob_name=imagefile)
                    prompt = config.get('prompt', '')
                    negPrompt = config.get('negative_prompt', '')
                    steps = config.get('steps', 50)
                    numImages = config.get('num_images', 8)
                    size = config.get('size', 360)
                    style = config.get('style', 'cartoon')
                    
                    # images = generate(prompt=prompt, negPrompt=negPrompt, image=imageData, steps=steps, numImages=numImages)
                    images = launchAigcScript(style, prompt, negPrompt, steps, numImages, imageData)
                    result = []
                    for image in images:
                        # d = image['base64_str']
                        d = image
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

        time.sleep(5)


app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    init()
    t = threading.Thread(target = aigcJobThread)
    t.start()

    web.run_app(app, port=8085)
