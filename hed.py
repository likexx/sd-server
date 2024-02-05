import cv2
import numpy as np
import random
from PIL import Image
import imageio

from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector

reader = imageio.get_reader('./input/dragonball.mp4', "ffmpeg")
frame_count = 30*6
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]


apply_hed = HEDdetector()

i = 0
j = 0
for img in pose_images:
    if i%4!=0:
        i+=1
        continue
    # img = Image.open('./input/pose_{}.png'.format(i))
    img = img.resize((512, 512))
    input_image = np.array(img)
    input_image = HWC3(input_image)
    detected_map = apply_hed(resize_image(input_image, 512))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, 512)
    H, W, C = img.shape
    # print(H, W, C)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    print(detected_map.shape)
    output_image = Image.fromarray(detected_map)
    output_image = output_image.resize((512, 512))
    output_image.save("./output/dragonball_hed_{}.png".format(j), 'PNG')
    i+=1
    j+=1

