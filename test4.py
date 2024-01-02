import aigc
import img_util

params = aigc.AigcParam(prompt="a girl and a body walking along on the street", style='sdxl', numImages=1, seed=10024)
client = aigc.AigcCPU(params)

images = client.generate()
i=2
for img in images:
    img_util.saveBase64toPNG(img['base64_data'], './output/{}.png'.format(i))
    i+=1
