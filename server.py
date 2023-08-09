from aiohttp import web
import torch
import base64, os
from io import BytesIO

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "123456")

def txt2img(prompt):
    img = pipe(prompt, num_inference_steps=10).images[0]
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # print(base64_str)
    return base64_str


routes = web.RouteTableDef()

@routes.post('/healthcheck')
async def handle(request):
    text = "ok, ai"
    return web.Response(text=text)


@routes.post('/txt2image')
async def text_to_image_handle(request):
    post = await request.json()
    prompt = post.get("prompt")
    if post.get('token') != SERVER_TOKEN:
        return web.json_response({'result': "invalid token"})

    data = txt2img(prompt)
    return web.json_response({'result': data})


app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app, port=8085)