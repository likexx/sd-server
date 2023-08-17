import requests
import json

URL_PREFIX = 'http://34.31.203.162:8080/api/v1'
headers = {
    'Content-type': 'application/json',
    'Authorization': 'token 63D4647F-00B6-49B3-BB40-6A1B927F843E',
    }

def getNextAvailableJob():
    url = URL_PREFIX + '/aigc/job/next'
    # JSON data you want to send
    data = {
    }

    response = requests.post(url, json=data, headers=headers).json()
    return response['data']


def updateJobStatus(jobId, status):
    url = URL_PREFIX + '/aigc/job/status/update'
    # JSON data you want to send
    data = {
        'job_id': jobId,
        'status': status
    }

    response = requests.post(url, json=data, headers=headers).json()
    return response['data']
    

def updateJobResult(jobId, result):
    url = URL_PREFIX + '/aigc/job/result/update'
    # JSON data you want to send
    data = {
        'job_id': jobId,
        'result': result
    }

    response = requests.post(url, json=data, headers=headers).json()
    return response['data']
    
