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
# jobId = "40f4f77d54ee45acb362acb72c282834"
# consumer.updateJobStatus(jobId, "failed")

# from datetime import datetime, timedelta
# from time import time, mktime
# import jwt, requests

# dt = datetime.now() + timedelta(minutes=24*60)

# header = {
# "alg": "ES256",
# "kid": "8S9VRA7DGF",
# "typ": "JWT"
# }

# payload = {
#   "iss": "69a6de78-b269-47e3-e053-5b8c7c11a4d1",
#   "iat": int(time()),
#   "exp": int(mktime(dt.timetuple())),
#   "aud": "appstoreconnect-v1",
# #   "bid": "com.like.prod.comicx"
# }
# with open('AuthKey_8S9VRA7DGF.p8', 'rb') as f:
#     secret = f.read()
# # print(secret)
# token = jwt.encode(payload, secret, algorithm="ES256", headers=header)
# # print(token)

# # decoded_token = token.decode('utf-8')
# # print(decoded_token)
# try:
#     headers = {'Authorization': f'Bearer {token}'}
#     print(headers)
#     r = requests.get("https://api.appstoreconnect.apple.com/v1/apps", headers=headers)
#     print(r.json())
#     # print(f"[R] {r.json()}")
# except Exception as e:
#     print(e)

import jwt
import requests
import time

# More details here
# https://developer.apple.com/documentation/appstoreconnectapi

def generate_token():
    exp_time = int(time.time() + 6000)
    payload = {
        "iss": "69a6de78-b269-47e3-e053-5b8c7c11a4d1",
        "iat": int(time.time()),
        "exp": exp_time,
        "aud": "appstoreconnect-v1"
    }
    headers = {
        "alg": "ES256",
        "kid": "8S9VRA7DGF",
        "typ": "JWT"
    }
    f = open("AuthKey_8S9VRA7DGF.p8", "r")
    private_key = f.read()
    token = jwt.encode(payload=payload, key=private_key, algorithm="ES256", headers=headers)
    return token

  
# Examples

# def delete_bundle_id(token: str, bundle_id: str):
#     r = requests.delete(f"https://api.appstoreconnect.apple.com/v1/bundleIds/{bundle_id}",
#                         headers={"Authorization": f"Bearer {token}"})
#     print(r.status_code)
#     print(r.text)


def get_bundle_ids(token: str):
    r = requests.get(f"https://api.appstoreconnect.apple.com/v1/bundleIds",
                     headers={"Authorization": f"Bearer {token}"})
    print(r.status_code)
    print(r.text)

token = generate_token()
get_bundle_ids(token)
