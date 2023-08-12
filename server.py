from aiohttp import web
import torch
import base64, os
from io import BytesIO
import argparse
from PIL import Image

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images) :
    return images, [False for i in images]

# edit StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc

pipeline = None
img2imgPipeline = None

SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "123456")
NUM_OF_IMAGES = 4
STEPS = 100
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
            pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                         torch_dtype=torch.float16, 
                                                         use_safetensors=True, 
                                                         variant="fp16", 
                                                         safety_checker = None, 
                                                         requires_safety_checker = False
                                                         )
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


def base64_to_rgb_image(base64_data):
    # Decode the base64 data
    decoded_data = base64.b64decode(base64_data)
    
    # Convert the decoded data to an image
    img_buffer = BytesIO(decoded_data)
    img = Image.open(img_buffer)
    
    # Convert to RGB
    rgb_img = img.convert('RGB')
    
    return rgb_img


def generate(prompt, 
            negPrompt = NEGATIVE_PROMPT, 
            image=None, 
            steps=50,
            numImages=NUM_OF_IMAGES
            ):
    global pipeline, img2imgPipeline

    if not image:
        images = pipeline(prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_images_per_prompt=NUM_OF_IMAGES,
                        num_inference_steps=STEPS,
                        height=HEIGHT,
                        width=WIDTH).images
    else:
        init_image = base64_to_rgb_image(image)
        init_image = init_image.resize((WIDTH, HEIGHT))
        images = img2imgPipeline(prompt,
                                image=init_image,
                                negative_prompt=NEGATIVE_PROMPT,
                                num_images_per_prompt=NUM_OF_IMAGES,
                                num_inference_steps=STEPS,
                                height=HEIGHT,
                                width=WIDTH).images
    result = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
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
    
    data = generate(prompt=prompt, negPrompt=negPrompt, image=image, steps=steps, numImages=num_images)
    return web.json_response({'result': data})

app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    init()

    web.run_app(app, port=8085)