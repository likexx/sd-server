from aiohttp import web
import torch
import base64, os, json, uuid, time, threading
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

pipeline = None
img2imgPipeline = None

pipeline_lock = threading.Lock()

WATERMARK_FONT = ImageFont.truetype("Arial.ttf", 30)

SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "123456")
NUM_OF_IMAGES = 4
HEIGHT=480
WIDTH=480
STEPS = 50

NEGATIVE_PROMPT="(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, ((bad hands)), text, error, missing fingers, extra digit, fewer digits,head out of frame, cropped, letterboxed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, letterbox, blurry, monochrome, fused clothes, nail polish, boring, extra legs, fused legs, missing legs, missing arms, extra arms, fused arms, missing limbs, mutated limbs, dead eyes, empty eyes, 2girls, multiple girls, 1boy, 2boys, multiple boys, multiple views, jpeg artifacts, text, signature, watermark, artist name, logo, low res background, low quality background, missing background, white background,deformed"

def initConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='base model path or name on huggingface')
    parser.add_argument('--lora', type=str, default='', help='lora location')
    parser.add_argument('--token', type=str, default="123456", help='server token')
    parser.add_argument('--steps', type=int, default=50, help='steps')
    parser.add_argument('--width', type=int, default=480, help='image width (same height)')
    # Parse the arguments
    args = parser.parse_args()
    return args


def init():
    global pipeline, img2imgPipeline, SERVER_TOKEN, STEPS, WIDTH, HEIGHT
    conf = initConfig()
    if conf.token != '':
        SERVER_TOKEN = conf.token

    STEPS = conf.steps
    WIDTH = conf.width
    HEIGHT = WIDTH

    model = conf.model
    loraPath = conf.lora
    print("model: {}\nlora:{}".format(model, loraPath))
    # pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
    if model.endswith('.safetensors') or model.endswith('.ckpt'):
        pipeline = StableDiffusionPipeline.from_single_file(model, safety_checker = None, requires_safety_checker = False)
    else:
        if model=='sdxl':
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

    if loraPath != '':
        if loraPath.endswith('.safetensors'):
            print("load lora weights")
            pipeline.load_lora_weights(".", weight_name=loraPath)
        else:
            pipeline.unet.load_attn_procs(conf.lora)

    pipeline.to("cuda")

def generate(prompt, 
            negPrompt = NEGATIVE_PROMPT, 
            image=None, 
            steps=50,
            numImages=NUM_OF_IMAGES
            ):
    global pipeline, img2imgPipeline, pipeline_lock

    result = []
    with pipeline_lock:
        if not image or not img2imgPipeline:
            images = pipeline(prompt,
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
        # print(base64_str)
    return result

routes = web.RouteTableDef()

@routes.post('/healthcheck')
async def handle(request):
    text = "ok, ai"
    return web.Response(text=text)


@routes.post('/txt2image')
async def text_to_image_handle(request):
    post = await request.json()
    if post.get('token') != SERVER_TOKEN:
        return web.json_response({'result': "invalid token"})

    prompt = post.get("prompt")
    negPrompt = post.get("neg_prompt")
    num_images = post.get("number_images")
    steps = post.get("steps")

    if num_images:
        num_images = int(num_images)
    else:
        num_images = NUM_OF_IMAGES

    if steps:
        steps = int(steps)
    else:
        steps = STEPS
    
    data = generate(prompt=prompt, negPrompt=negPrompt, steps=steps, numImages=num_images)
    return web.json_response({'result': data})

@routes.post('/img2image')
async def img_to_image_handle(request):
    post = await request.json()
    if post.get('token') != SERVER_TOKEN:
        return web.json_response({'result': "invalid token"})

    prompt = post.get("prompt")
    negPrompt = post.get("neg_prompt")
    image = post.get("image_data")
    num_images = post.get("number_images")
    steps = post.get("steps")

    if num_images:
        num_images = int(num_images)
    else:
        num_images = NUM_OF_IMAGES

    if steps:
        steps = int(steps)
    else:
        steps = STEPS
    
    data = generate(prompt=prompt, negPrompt=negPrompt, image=image, steps=steps, numImages=num_images)
    return web.json_response({'result': data})

def aigcJobThread():
    while True:
        try:
            job = consumer.getNextAvailableJob()
            if job:
                jobId = job['job_id']
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
                
                images = generate(prompt=prompt, negPrompt=negPrompt, image=imageData, steps=steps, numImages=numImages)
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
