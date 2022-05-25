import os
import pandas as pd
from ssoc_autocoder.processing import process_text
import json
import copy
from datetime import datetime
import boto3



def lambda_handler(event, context):
    
    s3 = boto3.resource('s3')
    
    #bucket name
    bucket ='mcf-job-id'

    buck=s3.Bucket(bucket)
    
    #iterating through all the files in raw_csv
    for s3_object in buck.objects.filter(Prefix='raw_csv/'):

      #get the filename without the path
      filename=s3_object.key.split("/")[-1]
      
      try:

         #downlad file into tmp
         buck.download_file('raw_csv/'+filename, f'/tmp/'+filename)
         
      except Exception as e:

         print(e)
         print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
         raise e 
    
    def cleaning_text_and_check(text):
        
        cleaned_text = process_text(text)
        
        # add in additional check for proper sentences
        
        return cleaned_text
        
    #lst is a list containing all the files in tmp directory
    lst = os.listdir("/tmp/")

    def output_individual_files():


        for filename in lst:
            #only selecting files beeginning with raw to avoid looping issue when processed gets added
            if filename.startswith('raw'):

                now = datetime.now()
                
                #read the raw_csv file into pandas
                df = pd.read_csv("/tmp/" + filename)

                #process the description column

                df['description']= df['description'].apply(cleaning_text_and_check)

                #tagging the time for the processed date
                df['Processed date'] = now

                #pandas to csv back into tmp directory after it has processed, it is saved with the prefix processed
                df.to_csv('/tmp/'+'processed'+filename[3:], index=False)
                    
    output_individual_files()
    
    try:
        
        #iteratring through the files in tmp
        for filename in os.listdir("/tmp/"):

            #only selecting files beeginning with raw
            if filename.startswith('processed'):
                
                #uploading the processed files to the processed_csv folder in the bucket
                s3.meta.client.upload_file(f'/tmp/{filename}', bucket, f'processed_csv/{filename}')
      
    except Exception as e:
        print(e)
    
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }