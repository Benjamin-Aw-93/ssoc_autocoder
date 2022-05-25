import json
import pandas as pd
import urllib.parse
import requests
import boto3
import os

# for trigger
#s3 = boto3.client('s3')
def lambda_handler(event, context):
    

   # Get the object from the event and show its content type (used for trigger)
   #bucket = event['Records'][0]['s3']['bucket']['name']

   #used for trigger
   #key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

   bucket ='mcf-job-id'


   key='MCF_Training_Set_Full.csv'
    
   try:
       #used for trigger 
        #response = s3.get_object(Bucket=bucket, Key=key)
        
        s3 = boto3.resource('s3')

        #downlad file from s3 bucket to tmp
        s3.meta.client.download_file('mcf-job-id', 'MCF_Training_Set_Full.csv', '/tmp/MCF_Training_Set_Full.csv')
        
        local_file_name = '/tmp/MCF_Training_Set_Full.csv'
        
        df=pd.read_csv(f'{local_file_name}')
        
   except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e

   #getting the Job_ID nd uuid columns
   df = df[['MCF_Job_Ad_ID', 'uuid']]

   #selecting the top 20 for testing purposes
   df = df.head(20)
   
   test = requests.get('https://api.mycareersfuture.gov.sg/v2/jobs/bb5faebc85f3504c17b83e16d2b4dafb')
    
   uuid_not_found = test.json()
   print(uuid_not_found)
   
   rate_limit_count = 0 
   errors = []
   
   base_url = 'https://api.mycareersfuture.gov.sg/v2/jobs'

   
   total_count = len(df)
      
   for i, ad_id, uuid in zip(list(range(1, len(df)+1)), df['MCF_Job_Ad_ID'], df['uuid']):
         
      
      
      req = requests.get(base_url + "/" + uuid)
      
      if req.status_code != 200:
         try:
               # if the uuid can't be found
               if req.json() == uuid_not_found:
                  errors.append(ad_id)
         except:
               # if we are getting rate limited
               print('Backing off...\r', end = '')
               rate_limit_count += 1
               time.sleep(2)
               req = requests.get(base_url + "/" + uuid)
               if req.status_code != 200:
                  errors.append(ad_id)
               
      if req.status_code == 200:
         try:                    
               with open(f'/tmp/{ad_id}.json', 'w') as file:
                  json.dump(req.json(), file)
                  
                  
                  
               #uploading from tmp to s3 bucket into the json folder  
               s3.meta.client.upload_file(f'/tmp/{ad_id}.json', bucket, f'json/{ad_id}.json')

               #print(f'{i}/{total_count} completed - called {ad_id} successfully! Error count: {len(errors)}, Rate limit count: {rate_limit_count}\r', end = '')
               
         except Exception as e:
               print(e)

   #to know that the function is completed
   return 'completed'