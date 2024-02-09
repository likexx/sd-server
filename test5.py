import aigc
import img_util

params = aigc.AigcParam(
    prompt= '''
likezhang, likezhang is sitting on the chair in the president room in the White House in US, {likezhang}, wearing fine suits in white, medium shot, very strong, US national flag on the wall, background details,
''',
    style='likezhang', 
    steps=60,
    numImages=4,
    seed=12345,
    imageWidth=1024,
    imageHeight=1024,
    deviceType='cuda')
# params.image = img_util.convert_image_to_base64('./input/p16.png')

workflow = aigc.AigcWorkflow(params)

images = workflow.generate()
i=33001
for img in images:
    imgBase64Data = img['base64_data']
    processedImage = img_util.add_watermark_to_base64(imgBase64Data, 'created by comicx.ai')
    d = img_util.image_to_base64(processedImage)    
    img_util.saveBase64toPNG(d, '/home/likezhang/output/likezhang_{}.png'.format(i))
    i+=1
