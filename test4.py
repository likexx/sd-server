import aigc
import img_util

params = aigc.AigcParam(
    # prompt="a man is killed by another soldier with a sword, his arm is cut off, his body is cut in half, blood is flooding the ground, view from aside, the killed man lies on the ground and covered with blood, the cut arm is by his body, the soldier is standing aside with a sword with blood", 
    prompt = "a human body is cut to half but still standing, all blood, the lower half of the body is standing, the upper half body is on the ground",
    # prompt = "a fresh human brain in a simple bag, no human face, no skull, no head, no skull, the brain is taken out from a human's head, the brain has some blood, realistic, photo realistic, details",
    style='anim-porn', 
    steps=30,
    numImages=2)
client = aigc.AigcCPU(params)

images = client.generate()
i=1801
for img in images:
    imgBase64Data = img['base64_data']
    processedImage = img_util.add_watermark_to_base64(imgBase64Data, 'created by comicx.ai')
    d = img_util.image_to_base64(processedImage)    
    img_util.saveBase64toPNG(d, './output/{}.png'.format(i))
    i+=1
