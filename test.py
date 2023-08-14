import img_util as imgUtil
from PIL import Image
import consumer
import json
# img = Image.open('g2.png')
# img2 = imgUtil.add_watermark(img, "Created by KK Studio")
# img2.save("result.png")

# job = consumer.getNextAvailableJob()
# print(job)

# if job is None:
#     exit(0)

# jobId = job['job_id']
jobId = "40f4f77d54ee45acb362acb72c282834"
consumer.updateJobStatus(jobId)

