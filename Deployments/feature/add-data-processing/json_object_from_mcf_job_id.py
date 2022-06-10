import json
import boto3
import hashlib
import urllib3
import requests

http = urllib3.PoolManager()

def lambda_handler(event, context):
    
    s3 = boto3.resource('s3')
    
    #query parameter
    mcf_job_id = event['queryStringParameters']['jobId']
    
    #get the uuid
    uuid = hashlib.md5(mcf_job_id.encode()).hexdigest()
    
    
    #bucket the json object needs to be in 
    dir_bucket ='mcf-job-id-json'
    
    #base url for the api
    base_url = 'https://api.mycareersfuture.gov.sg/v2/jobs'

    
    #check if the job id is already in the bucket
    try:
            
        s3.meta.client.download_file(dir_bucket, f'{mcf_job_id}.json', '/tmp/'+f'{mcf_job_id}.json')
        
        response = {
        'statusCode': 400,
        'body': json.dumps(f" Already have {mcf_job_id}.json in {dir_bucket} bukcet ")
    }
        return response
        
    except:
        
        pass
    
    #api request
    req = requests.get(base_url + "/" + uuid)
  
    #if unsucessful
    if req.status_code != 200:
        response = {
            'statusCode': 400,
            'body': json.dumps({'Error': 'Error calling the MCF Jobs API with the provided MCF job ad ID. Please check that you have a valid MCF job ad ID and call this API again.'})
        }
        return response
    
    #dump the json object ito the bucket
    else:
        
        
                         
        try:                    
            with open(f'/tmp/{mcf_job_id}.json', 'w') as file:
                json.dump(req.json(), file)
                
                
            s3.meta.client.upload_file(f'/tmp/{mcf_job_id}.json', dir_bucket, f'{mcf_job_id}.json')
            
        except Exception as e:
            
            response = {
            'statusCode': 400,
            'body': json.dumps(f'{e}')
        }
            return response
    
    #return successful response 
    response = {
        'statusCode': 200,
        'body': json.dumps(f' Successfully put {mcf_job_id}.json in {dir_bucket} bukcet')
    }
            
    return response