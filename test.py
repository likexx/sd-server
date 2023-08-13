import img_util as imgUtil
from PIL import Image

img = Image.open('g2.png')
img2 = imgUtil.add_watermark(img, "作者：工作室")
img2.save("result.png")