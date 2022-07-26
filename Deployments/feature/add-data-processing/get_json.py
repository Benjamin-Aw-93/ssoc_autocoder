import json
import pandas as pd
import urllib.parse
import requests
import boto3
import os
from datetime import date, datetime, timedelta
import time

def get_object(file_name,yesterday):
    """
    Checks if the the files are the latest files 

    Parameters:
        file_name (str): file name that you want to check about
        yesterday (date): Date to input 

    Returns:
        True or False depending if it is modified
        within the period between today and yesterday
    """
    
    try:
        obj = file_name.get(IfModifiedSince=yesterday)
        return obj['Body']
    except:
        False
        
def read_bucket(buck,path,x):
    
    s3 = boto3.resource('s3')
    try:
       
        

        #downlad file from s3 bucket to tmp
        s3.meta.client.download_file(buck, x, path+f'{x}')
        
        local_file_name = f'{path}{x}'
        
        df=pd.read_csv(f'{local_file_name}')
        
        return df
        
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(x, buck))
        raise e
        

    
        
def task(df,dir_bucket,path):
    """
    Main driver function 

    Parameters:
        x (str) : filename 
    Returns:
        Does not resturn anything processes the api
        to json objects and store them in the s3 bucket
    """
    
    s3 = boto3.resource('s3')
    
    

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
                with open(f'{path}{ad_id}.json', 'w') as file:
                    json.dump(req.json(), file)
               
               #uploading from tmp to s3 bucket into the json folder  
                
                s3.meta.client.upload_file(f'{path}{ad_id}.json', dir_bucket, f'{ad_id}.json')
      
            except Exception as e:
                print(e)

   #to know that the function is completed
    print('went through')
    
def lambda_handler(event, context):
    
    bucket ='mcf-job-id-csv'
    key='MCF_Training_Set_Full.csv'
    
    s3 = boto3.resource('s3')
    
    #input number of days to check since when has it been modified
    time = 15
    
    #benchmark of date to check since when it has last been modified
    yesterday = datetime.fromisoformat(str(date.today() - timedelta(days=time)))
    
    
    buck=s3.Bucket(bucket)
    
    
        
    for s3_object in buck.objects.all():
        
        s3_file = get_object(s3_object,yesterday)
        print(s3_file)
        if s3_file:
            df = read_bucket(bucket,'/tmp/',s3_object.key)
            task(df,'mcf-job-id-json')
        else:
            print(f'{s3_object.key} is a old file')
    return 'completed'