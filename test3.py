import requests, time, json, jwt

def testInappReceipts():
    try:

        KEY_ID = '8S9VRA7DGF'
        #ISSUER_ID = INAPP_ISSUER_ID
        #KEY_ID = '837666KJ65'
        ISSUER_ID = '69a6de78-b269-47e3-e053-5b8c7c11a4d1'
        # START_TIME = int(round(time.time()) ) # 20 minutes timestamp
        # EXPIRATION_TIME = int(round(time.time() + (24 * 60.0 * 60.0))) # 20 minutes timestamp
        START_TIME = round(time.time()) # 20 minutes timestamp
        EXPIRATION_TIME = round(time.time() + (2 * 60.0 * 60.0)) # 20 minutes timestamp
        print("START_TIME ---> " , START_TIME)

        PATH_TO_KEY = 'AuthKey_8S9VRA7DGF.p8'
        #PATH_TO_KEY = './appStoreConnect_apikey.p8'
        with open(PATH_TO_KEY, 'rb') as f:
            PRIVATE_KEY = f.read()

        header = {
            "alg": "ES256",
            "kid": KEY_ID,
            "typ": "JWT"
        }

        payload = {
            "iss": ISSUER_ID,
             "iat": START_TIME,
             "exp": EXPIRATION_TIME,
             "aud": "appstoreconnect-v1",
             "bid": "com.like.prod.comicx",
        }

        # options={
        #     'verify_exp': False,  # Skipping expiration date check
        #     'verify_aud': False 
        # }

        # Create the JWT
        token = jwt.encode( payload, PRIVATE_KEY ,headers=header )
        print("token is " , token)
        print("PRIVATE_KEY" ,PRIVATE_KEY )
        # token = jwt.encode(header, payload, PRIVATE_KEY, options=options)
        original_transaction_id = '2000000424851043'
        # API Request
        #JWT = 'Bearer ' + token.decode()
        JWT = 'Bearer ' + token
        print("JWT ---> " , JWT)
        URL = f'https://api.storekit-sandbox.itunes.apple.com/inApps/v1/transactions/{original_transaction_id}'
        # URL = f'https://api.appstoreconnect.apple.com/v1/users/'
        HEAD = {'Authorization': JWT}

        print("URL--> ", URL)
        print("hEAD--> ", HEAD)
        r = requests.get(URL, params={'limit': 200}, headers=HEAD)
        print("r--->" , r)
        #print("output from test --> " , json.dumps(r.json()) )

        # Write the response in a pretty printed JSON file
        # with open('output.json', 'w') as out:
        #     out.write(json.dumps(r.json(), indent=4))
        # response = dict(status="success", message="Succeed")
        # logger.info("Succeed")
        # return response, HTTPStatus.BAD_REQUEST
            
    except:
        print("Failed")
        

testInappReceipts()