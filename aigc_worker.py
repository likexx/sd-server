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

import aigc

WATERMARK_FONT = ImageFont.truetype("Arial.ttf", 30)

SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "123456")

def worker():
    try:
        job = consumer.getNextAvailableJob()
        if not job:
            print("no job found")
        if job:
            # print(job)
            jobId = job['job_id']
            try:
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

                watermark = config.get('watermark', True)

                if style == 'realistic':
                    style = 'real'
                elif style == 'realistic-adult':
                    style = 'real-adult'

                consumer.updateJobStatus(jobId, "generating")
                print(job)
                if style != "dalle3":
                    # images = generate(prompt=prompt, 
                    #                 negPrompt=negPrompt, 
                    #                 image=imageData, 
                    #                 steps=steps, 
                    #                 numImages=numImages, 
                    #                 style=style)
                    params = aigc.AigcParam(
                         prompt=prompt,
                         image = imageData,
                         steps = steps,
                         numImages= numImages,
                         style = style
                    )
                    client = aigc.AigcCPU(params)
                    images = client.generate()                    
                else:
                    images = imgUtil.generate_with_dalle3(prompt=prompt, k=numImages)


                result = []
                for image in images:
                    if not image['base64_data']:
                        continue
                    imgBase64Data = image['base64_data']
                    d = imgBase64Data
                    if watermark == True:
                        processedImage = imgUtil.add_watermark_to_base64(imgBase64Data, 'created by comicx.ai')
                        d = imgUtil.image_to_base64(processedImage)

                    aigcBucketName = bucket.aigc_img_bucket_name
                    aigcFilename = jobId + '-' + str(uuid.uuid4()).replace('-', '')
                    bucket.upload_to_bucket(d, aigcBucketName, aigcFilename)
                    imgUtil.saveBase64toPNG(d, './output/{}_{}.png'.format(jobId, aigcFilename))
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
