import json
import pandas as pd
import urllib.parse
import requests
import boto3
import os
from datetime import date, datetime, timedelta


def lambda_handler(event, context):

   s3 = boto3.resource('s3')
   bucket = 'mcf-job-id'
   buck=s3.Bucket(bucket)


   key='MCF_Training_Set_Full.csv'

   #input number of days to check since when has it been modified
   time = 1
    
    #benchmark of date to check since when it has last been modified
   yesterday = datetime.fromisoformat(str(date.today() - timedelta(days=time)))

   #functionn to check if the file is one of the latest files
   def get_object(file_name):

      try:
         obj = file_name.get(IfModifiedSince=yesterday)
         return obj['Body']
      except:
         False

   # the main driver function converting the job ID to json objects and putting them in a bucket 
   def task(filename):
      try:
         
         s3 = boto3.resource('s3')

         #downlad file from s3 bucket to tmp
         s3.meta.client.download_file('mcf-job-id', filename, '/tmp/MCF_Training_Set_Full.csv')
         
         local_file_name = '/tmp/{filename}'
         
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
         
            except Exception as e:
                  print(e)

      #to know that the function is completed
      print(f'{filename} went through')

   for s3_object in buck.objects.all():
      s3_file = get_object(s3_object)

      # if the file is the one of the latest files
      if s3_file:
         task(s3_object.key)
      else:
         print(f'{s3_object.key} is a old file')

   return 'completed'