import requests
import json, os

URL_PREFIX_LIST = [
    # 'http://34.31.203.162:8080/api/v1',
    # 'http://10.128.0.18:8080/api/v1',
    'http://35.184.169.61:8080/api/v1',
]

URL_PREFIX = 'http://34.31.203.162:8080/api/v1'

def createHeaders():
    headers = {
        'Content-type': 'application/json',
        'Authorization': 'token 63D4647F-00B6-49B3-BB40-6A1B927F843E',
        }
    
    envToken = os.getenv("AIGC_ACCESS_TOKEN")
    if envToken:
        headers['Authorization'] = "token {}".format(envToken)
    return headers

def getNextAvailableJob():
    global URL_PREFIX, URL_PREFIX_LIST
    for prefix in URL_PREFIX_LIST:
        URL_PREFIX = prefix
        url = URL_PREFIX + '/aigc/job/next'
        # JSON data you want to send
        data = {
        }
        print("fetch job from {}".format(url))
        response = requests.post(url, json=data, headers=createHeaders()).json()
        jobData = response['data']
        if jobData:
            return jobData
    return None

def updateJobStatus(jobId, status):
    url = URL_PREFIX + '/aigc/job/status/update'
    # JSON data you want to send
    data = {
        'job_id': jobId,
        'status': status
    }

    response = requests.post(url, json=data, headers=createHeaders()).json()
    return response['data']
    

def updateJobResult(jobId, result):
    url = URL_PREFIX + '/aigc/job/result/update'
    # JSON data you want to send
    data = {
        'job_id': jobId,
        'result': result
    }

    try:
        response = requests.post(url, json=data, headers=createHeaders()).json()
        print(response)
        return response['data']
    except Exception as err:
        print("update job result error: {}".format(err))
        return True
    
