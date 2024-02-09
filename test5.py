import aigc
import img_util

params = aigc.AigcParam(
    prompt= '''
likezhang is standing on the ground, wearing jeans and t-shirt, waving his hands for hello
''',
    style='likezhang', 
    steps=100,
    numImages=4,
    seed=0,
    deviceType='cuda')
# params.image = img_util.convert_image_to_base64('./input/p16.png')

workflow = aigc.AigcWorkflow(params)

images = workflow.generate()
i=33001
for img in images:
    imgBase64Data = img['base64_data']
    processedImage = img_util.add_watermark_to_base64(imgBase64Data, 'created by comicx.ai')
    d = img_util.image_to_base64(processedImage)    
    img_util.saveBase64toPNG(d, './output/likezhang_{}.png'.format(i))
    i+=1
