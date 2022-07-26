import json
import pandas as pd
import urllib.parse
import requests
import boto3
import os
from datetime import datetime

def lambda_handler(event, context):
    
   s3 = boto3.resource('s3')

   #bucket name
   bucket ='mcf-job-id'

   buck=s3.Bucket(bucket)

   #iterrating through all the files in json folder of the s3 bucket
   for s3_object in buck.objects.filter(Prefix='json/'):

      #get the filename without the path
      filename=s3_object.key.split("/")[-1]
      
      try:

         #downlad file into tmp
         buck.download_file('json/'+filename, f'/tmp/'+filename)
      except Exception as e:

         print(e)
         print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
         raise e
   
   def extract_mcf_data(json):
   
      output = {}
      transfer = ['uuid', 'title', 'description', 'minimumYearsExperience', 'numberOfVacancies']
      # Extracting general information of the job posting
      for key in transfer:
         try:
               output[key] = json[key]
         except:
               # If keys not found, treat file as failure to extract
               return None, None
   
      # Extract skills, skills are mainly captured in separate JSON objects 
      output['skills'] = ', '.join([entry['skill'] for entry in json['skills']])
      
      # Extract hiring company
      company = ['name', 'description', 'ssicCode', 'employeeCount']
      if json['metadata']['isPostedOnBehalf']:
         company_col = 'hiringCompany'
      else:
         company_col = 'postedCompany'
      for key in company:
         try:
               output['company_' + key] = json[company_col][key]
         except TypeError:
               output['company_' + key] = json[company_col]
         
      # Extract additional infomation such as the date of the post, number of views and applications etc
      metadata = ['originalPostingDate', 'newPostingDate', 'expiryDate', 'totalNumberOfView', 'totalNumberJobApplication']
      for key in metadata:
         output[key] = json['metadata'][key]
      
      # Extract salary, if min and max is not available, return None which is captured in the except statement
      salary = ['maximum', 'minimum']
      for key in salary:
         try:
               output['salary_' + key] = json['salary'][key]
         except TypeError:
               output['salary_' + key] = json['salary']
      
      # Extract additional salary information
      try:
         output['salary_type_id'] = json['salary']['type']['id']
         output['salary_type'] = json['salary']['type']['salaryType']
      except TypeError:
         output['salary_type_id'] = json['salary']
         output['salary_type'] = json['salary']
         
      # Return the actual output, and the date of the post       
      return output, output['originalPostingDate']
      # TODO implement
   def extract_and_split():
   
      output = {}

      #to ge the time
      now = datetime.now()
      
      
   
      for filename in os.listdir("/tmp/"):    
         
         print(f'Reading in {filename}')
         f = open("/tmp/"+ filename)
         entry = json.load(f)
         
         extracted_result, date = extract_mcf_data(entry)
         
   
         if extracted_result:

               #to include the column for the Job_Ad_Id
               extracted_result["MCF_Job_Ad_ID"] = filename[:-5]

               #tag the time it was processed
               extracted_result['JSON to CSV date'] = now
               date_year_mth = date[0:7]
               if date_year_mth in output: 
                  output[date_year_mth].append(extracted_result)
               else:
                  output[date_year_mth] = [extracted_result]
         else:
               print(f'{filename} has missing key values')
               
      
      return output
   def write_to_csv(output):
   
      for dates in output.keys():
         pd.DataFrame(output[dates]).to_csv("/tmp/raw_" + dates + ".csv", index = False)
   
   output = extract_and_split()
   write_to_csv(output)
   

   #iteratring through the files in tmp
   for filename in os.listdir("/tmp/"):

      #only selecting files beeginning with raw
      if filename.startswith('raw'):
        
        try:

         #upload from tmp to s3
         s3.meta.client.upload_file(f'/tmp/{filename}', bucket, f'raw_csv/{filename}')

        except Exception as e:
            
            print(e)

   
   # to test if it has ran successfully 
   return {
      'statusCode': 200,
      'body': json.dumps('Hello from Lambda!')
   }