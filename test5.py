import aigc
import img_util

params = aigc.AigcParam(
    prompt= '''
likezhang is sitting behind a table in the president room in the White House in US. {{likezhang}} is looking at you. {likezhang}, 1man, 1 man, close up shot, view from front, face details, background details
''',
    style='likezhang', 
    steps=60,
    numImages=8,
    seed=0,
    deviceType='cuda',
    weightedPrompt=False)
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
