import requests, time, json
from authlib.jose import jwt

KEY_ID = "8S9VRA7DGF"
ISSUER_ID = "69a6de78-b269-47e3-e053-5b8c7c11a4d1"
EXPIRATION_TIME = int(round(time.time() + (100 * 12 * 30 * 24* 60 * 60.0))) # 20 minutes timestamp
PATH_TO_KEY = 'AuthKey_8S9VRA7DGF.p8'
with open(PATH_TO_KEY, 'r') as f:
    PRIVATE_KEY = f.read()

header = {
    "alg": "ES256",
    "kid": KEY_ID,
    "typ": "JWT"
}

payload = {
    "iss": ISSUER_ID,
    "exp": EXPIRATION_TIME,
    "aud": "appstoreconnect-v1",
    "bid": "com.like.prod.comicx",
}

# Create the JWT
token = jwt.encode(header, payload, PRIVATE_KEY)

# API Request
JWT = 'Bearer ' + token.decode()
URL = 'https://api.appstoreconnect.apple.com/v1/users'
HEAD = {'Authorization': JWT}

print(JWT)
print('\n')
r = requests.get(URL, params={'limit': 200}, headers=HEAD)
print(r.json())
# Write the response in a pretty printed JSON file
# with open('output.json', 'w') as out:
#     out.write(json.dumps(r.json(), indent=4))