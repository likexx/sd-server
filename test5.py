import aigc
import img_util

params = aigc.AigcParam(
    prompt= 'meexxi and likezhang',
    style='likezhang', 
    steps=50,
    numImages=8,
    seed=0,
    deviceType='cuda',
    weightedPrompt=True)
# params.image = img_util.convert_image_to_base64('/home/likezhang/output/ref_1.png')

workflow = aigc.AigcWorkflow(params)

images = workflow.generate()
i=1
for img in images:
    imgBase64Data = img['base64_data']
    processedImage = img_util.add_watermark_to_base64(imgBase64Data, 'created by comicx.ai')
    d = img_util.image_to_base64(processedImage)    
    img_util.saveBase64toPNG(d, '/home/likezhang/output/likezhang_{}.png'.format(i))
    i+=1

# from diffusers import AutoPipelineForText2Image
# import torch

# pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# pipeline.load_lora_weights("/home/likezhang/output", weight_name="likezhang.safetensors")
# images = pipeline(prompt="likezhang is sitting on a chair and looking forward, {likezhang}, portrait, close up, view from front", num_images_per_prompt=4).images
# i=1
# for img in images:
#     img.save("/home/likezhang/output/likezhang_{}.png".format(i))
#     i+=1

# from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
# import torch

# pipeline = StableDiffusionPipeline.from_pretrained("/home/likezhang/output", torch_dtype=torch.float16).to("cuda")

# images = pipeline(prompt="meexxi and likezhang", num_images_per_prompt=4).images
# i=1
# for img in images:
#     img.save("/home/likezhang/output/all_{}.png".format(i))
#     i+=1
