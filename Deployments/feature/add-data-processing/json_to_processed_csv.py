import json
import os
import pandas as pd
from ssoc_autocoder.processing import process_text
from ssoc_autocoder.converting_json import extract_and_split,extract_mcf_data
import copy
from datetime import date, datetime, timedelta
import requests
import boto3





def lambda_handler(event, context):

    path = '/tmp/'
    
    ##input number of days to check since when has it been modified
    time = 1
    
    #benchmark of date to check since when it has last been modified
    yesterday = datetime.fromisoformat(str(date.today() - timedelta(days=time)))
    
    s3 = boto3.resource('s3')
    bucket ='raw-json-mcf'
    buck=s3.Bucket(bucket)
    
    #function for checking if the file is up to date based on the yesterday date
    def get_object(file_name):
        try:
            obj = file_name.get(IfModifiedSince=yesterday)
            return obj['Body']
        except:
            False
            
    for s3_object in buck.objects.all():
        
        filename=s3_object.key
        
        # to check if the file is the latest added file
        if get_object(s3_object):
        
            try:

                #downlad the file into tmp
                buck.download_file(filename, f'/tmp/'+filename)

            except Exception as e:
                print(e)
                print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
                raise e
        else:
            
            print(f'{filename} is too old')
      
    
    def write_to_csv(output):
    
        for dates in output.keys():
            pd.DataFrame(output[dates]).to_csv("/tmp/raw_" + dates + ".csv", index = False)
    
    def cleaning_text_and_check(text):
        
        cleaned_text = process_text(text)
        
        # add in additional check for proper sentences
        
        return cleaned_text


    def output_individual_files(path):

        lst = os.listdir(path)
        for filename in lst:
            if filename.startswith('raw'):
                
                now = datetime.now()
                
                df = pd.read_csv(path + filename)
                df['description']= df['description'].apply(cleaning_text_and_check)
                df['Processed date'] = now
                df.to_csv(path+'processed'+filename[3:], index=False)

    
    output = extract_and_split(path)
    write_to_csv(output)
    output_individual_files(path)

    directed_buck = 'weekly-csv-mcf'

    try:
        
        #iteratring through the files in tmp
        for filename in os.listdir("/tmp/"):
            #only selecting files beeginning with raw
            if filename.startswith('processed'):
                
                s3.meta.client.upload_file(f'/tmp/{filename}', directed_buck, f'{filename}')
      
    except Exception as e:
        print(e)
    
    # signal that it is completed
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }