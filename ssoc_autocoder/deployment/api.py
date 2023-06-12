import logging
logger = logging.getLogger('autocoder')

# Importing required libraries
import json
import random
import hashlib
import re
import pickle
import os
import zipfile
import time
import requests
import urllib.request

# Importing the deep learning libraries
from transformers import DistilBertTokenizer
import torch

# Initialising the HTTP object for the API call
import urllib3
http = urllib3.PoolManager()

# Importing the FastAPI functions
from fastapi import FastAPI, HTTPException, Query, status

# Importing our custom scripts
import ssoc_autocoder.model_prediction as model_prediction
import ssoc_autocoder.model_training as model_training
import ssoc_autocoder.data.text_cleaning as text_cleaning
import ssoc_autocoder.utils as utils

# Importing boto functions
import boto3
from botocore.exceptions import ClientError

# Function for initialise the required objects for prediction
def initialise():

    logger.info("Loading the artifacts needed for model prediction...")

    # If this container is running in the CStack environment
    if os.getenv("namespace") is not None:

        logger.info(f'Detected that container is in CStack namespace {os.getenv("namespace")}')
        
        # If we are in the dev environment, we download from CStack-provisioned S3 bucket
        if os.getenv("namespace").split('-')[3] == 'dev':
            
            logger.info(f'Currently in namespace dev')
            
            # Initialise S3 resource
            try:
                logger.debug(f'Trying to connect to S3 resource')
                s3 = boto3.resource('s3')
                logger.debug(f'Connected to S3 resource')
            except ClientError as e:
                logger.debug(f'Error in importing')
                logger.error(e)

            # Download file from specified S3 bucket - note that we need to change the bucket name depending on the environment
            s3_bucket_name = "t-dev-autocoder-artifacts"
            logger.info(f'Downloading artifact.zip')
            s3.Bucket(s3_bucket_name).download_file("artifacts.zip", "artifacts.zip")
            logger.info(f'Downloaded the artifacts from the S3 bucket {s3_bucket_name}')

        # If we are in the staging or production environment, we download from our own S3 bucket instead
        elif (os.getenv("namespace").split('-')[3] == 'stg') or (os.getenv("namespace").split('-')[3] == 'prd'):

            # Make the API call
            resp = requests.get("https://k3czta3yq6.execute-api.us-east-1.amazonaws.com/default/accessS3bucket",
                                headers = {"X-API-Key": "B4KjFTDCsv2yfB83eAlnG4XBk2SW8Ukq7eAUpT71"})

            # Extract the download URL
            download_url = resp.json()['url']

            # Download the file to the local folder
            urllib.request.urlretrieve(download_url, 'artifacts.zip')

        # If anything else, raise an error
        else:
            raise AssertionError("Error: namespace does not belong to any of the dev, stg, or prd environments")

        # Unzip the file
        with zipfile.ZipFile('artifacts.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        logger.debug(f'Unzipped the artifacts into the local directory')

    # If this is being tested locally (in Python virtual environment)
    else:    

        logger.info(f'Detected that container is in the local environment')

    # Load artifacts
    model_filepath = 'artifacts/ssoc-autocoder-model.pickle'
    tokenizer_filepath = 'artifacts/distilbert-tokenizer-pretrained-7epoch'
    ssoc_idx_encoding_filepath = 'artifacts/ssoc-idx-encoding.json'
    job_title_mapping = "artifacts/WPD_Job_title_list_mapping.json"
    ssoc_desc = "artifacts/ssoc_desc.json"

    # Reading in the SSOC-index encoding
    encoding = model_training.import_ssoc_idx_encoding(ssoc_idx_encoding_filepath)
    logger.debug('Loaded SSOC index encoding')

    # Reading in the SSOC descriptions
    with open(ssoc_desc, 'rb') as json_file:
        ssoc_desc = json.load(json_file)
    logger.debug('Loaded SSOC descriptions')

    # Load the model and tokenizer objects
    with open(model_filepath, 'rb') as handle:
        model = pickle.load(handle)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_filepath)
    logger.info(f'Successfully loaded all required artifacts!')

    # Load in WPD job titles list 
    with open(job_title_mapping) as mapping_file:
        job_title_map = json.load(mapping_file)
    logger.debug('Loaded WPD job title mapping')
    
    return model, tokenizer, encoding, ssoc_desc, job_title_map

def convert_to_uuid(input, input_type):
    """
    Converts the MyCareersFuture job ad ID or URL into a UUID
    """

    # If we are converting the MCF job ad ID to UUID
    if input_type == "id":

        # Validating the job ad ID
        if type(input) != str:
            raise HTTPException(status_code = 404, 
                                detail = "MCF job ad ID needs to be entered as a string.")
        elif (input[0:4] != "MCF-") and (input[0:4] != "JOB-"):
            raise HTTPException(status_code = 404, 
                                detail = "Invalid MCF job ad ID provided.")

        # If checks pass, then hash the job ad ID
        mcf_uuid = hashlib.md5(input.encode()).hexdigest()
        logger.info(f'Converted MCF job ad ID "{input}" to "{mcf_uuid}"')

    # If we are converting the URL to UUID
    elif input_type == "url":

        # Checking data type of the job ad URL
        if type(input) != str:
            logger.warning(f'MCF job ad URL "{input}" is not entered as a string.')
            raise HTTPException(status_code = 404, 
                                detail = "MCF job ad URL needs to be entered as a string.")

        # Note that we append a "?" to the end of the URL to ensure we capture the ID correctly
        regex_matches = re.search('\\-{1}([a-z0-9]{32})\\?', input + "?")

        # Try extracting the MCF job ad ID from the URL
        try:
            mcf_uuid = regex_matches.group(1)
            logger.info(f'Identified MCF job ad UUID: {mcf_uuid}')
        except:
            logger.warning('Could not identify hashed MCF job ad ID from the URL.')
            raise HTTPException(status_code = 404, 
                                detail = "Invalid URL from MyCareersFuture. Please check that you have a valid URL and call this API again.")

    # If we are picking a random UUID
    elif input_type == 'feelinglucky':

        # Reading in the dummy data
        with open('artifacts/dummy_data.json') as json_file:
            dummy_data = json.load(json_file)

        # Select one of the dummy data randomly
        data = dummy_data[random.randint(0,49)]

        # Retrieve the MCF job ad ID and overwriting it with one of our samples
        mcf_job_ad_id = data['MCF_Job_Ad_ID']
        mcf_uuid = hashlib.md5(mcf_job_ad_id.encode()).hexdigest()

        logger.info(f'Selected random MCF job ad ID "{mcf_uuid}"')

    return mcf_uuid

def call_mcf_api(mcf_uuid):
    """
    Calls the MCF jobs API with the provided UUID, with
    in-built error handling.
    """

    # Call the MCF API
    resp = http.request('GET', f'https://api.mycareersfuture.gov.sg/v2/jobs/{mcf_uuid}')

    # Error handling if an invalid MCF job ad ID is provided
    if json.loads(resp.data) == {'message': 'UUID is not found in the database.'}:

        logger.info("Invalid MCF job ad ID provided.")
        raise HTTPException(status_code = 404, 
                            detail = "Error finding the MyCareersFuture job ad specified. Please check that you have a valid MyCareersFuture job ad and try again.")
 
    # Error handling for all other potential errors
    elif resp.status != 200:
        
        logger.warning('Non-200 status when calling MCF jobs API. Trying again...')

        # Try again one more time after waiting for a split second
        time.sleep(0.1)
        resp = http.request('GET', f'https://api.mycareersfuture.gov.sg/v2/jobs/{mcf_uuid}')

        # If it fails again, we return a 500 error
        if resp.status != 200:
            logger.warning("Non-200 status when calling MCF jobs API. Returning status code of 500")
            raise HTTPException(status_code = 500, 
                                detail = "Error in fetching job ad from MyCareersFuture. Please try again later.")
 

    logger.info(f'MCF jobs API called successfully for job ad UUID {mcf_uuid}')

    # Extracting the key details of the job ad that we need
    job_id = json.loads(resp.data)['metadata']['jobPostId']
    job_title = json.loads(resp.data)['title']
    job_desc = json.loads(resp.data)['description']
    
    return job_id, job_title, job_desc

def generate_embeddings(model, 
                        tokenizer, 
                        job_title,
                        job_desc):
    
    """
    Generates the embeddings for the job ad.
    """
    
    embeddings = model_prediction.generate_embeddings(model, tokenizer, job_title, text_cleaning.process_text(job_desc))

    logger.debug('Prediction generated')

    return embeddings

def generate_predictions(model, tokenizer, encoding, ssoc_desc, job_title, job_desc, job_title_map, n_results, return_occupation, return_confidence_score, return_description, job_id = None):
    """
    Generates and formats the predictions for output.
    """
    predictions = model_prediction.generate_single_prediction(model, 
                                                              tokenizer,
                                                              job_title,
                                                              text_cleaning.process_text(job_desc),
                                                              None,
                                                              encoding,
                                                              n_results,
                                                              return_occupation, 
                                                              return_confidence_score, 
                                                              return_description,
                                                              )['SSOC_5D']
    logger.debug('Prediction generated')

    predictions_formatted = []
    
    for i in range(0, n_results):
        prediction_formatted = {
            'SSOC_Code': str(predictions['predicted_ssoc'][i]),
            'SSOC_Title': ssoc_desc[str(predictions['predicted_ssoc'][i])]['title']
        }
        if return_description:
            prediction_formatted['SSOC_Description'] =  ssoc_desc[str(predictions['predicted_ssoc'][i])]['description']
        if return_confidence_score:
            prediction_formatted['Prediction_Confidence'] =  f"{predictions['predicted_proba'][i]*100:.2f}%"        
        if return_occupation:
            prediction_formatted['Occupations'] = utils.wpd_job_title_converter(prediction_formatted['SSOC_Code'], job_title_map)
        
        predictions_formatted.append(prediction_formatted)

    logger.debug('Formatted predictions correctly for output')
    
    # Initialise output with the MCF information
    output = {
        'job_id': job_id,
        'job_title': job_title,
        'job_desc': job_desc,
        'top_prediction': predictions_formatted[0],
        'other_predictions': predictions_formatted[1:n_results]
    }

    return output

def validate_parameters(query_type, 
                        job_title,
                        job_desc,
                        id,
                        url,
                        n_results):

    if query_type == "text":
        if (job_title is not None) and (job_desc is not None):
            return True
        elif job_title is not None:
            raise HTTPException(status_code = 400, 
                                detail = "Query type selected is 'text', but no job title was provided.")
        elif job_desc is not None:
            raise HTTPException(status_code = 406, 
                                detail = "Query type selected is 'text', but no job description was provided.")
        else:
            raise HTTPException(status_code = 406, 
                                detail = "Query type selected is 'text', but no job title nor job description was provided.")
    elif query_type == "id":
        if id is not None:
            return True
        else:
            raise HTTPException(status_code = 406, 
                                detail = "Query type selected is 'id', but no MyCareersFuture job ad ID was provided.")
    elif query_type == "url":
        if url is not None:
            return True
        else:
            raise HTTPException(status_code = 406, 
                                detail = "Query type selected is 'url', but no MyCareersFuture job ad URL was provided.")
    
    if n_results < 30:
        return True
    else: 
        raise HTTPException(status_code = 416,
                            detail = f"Model is expected to return up to 30 results, n_results is currently set at {n_results}")

  
# def return_uuid(mcf_url: str):
#     """
#     Extracts the MCF UUID from the provided URL.
#     """

#     print(f"> Extracting MCF UUID from '{mcf_url}'")

#     # Check if the user wants to randomly grab a previous query
#     if mcf_url == 'feelinglucky':
        
#         print(">> User wants a random previous query")

#         # Reading in the dummy data
#         with open('artifacts/dummy_data.json') as json_file:
#             dummy_data = json.load(json_file)

#         # Select one of the dummy data randomly
#         data = dummy_data[random.randint(0,49)]

#         # Retrieve the MCF job ad ID and overwriting it with one of our samples
#         mcf_job_ad_id = data['MCF_Job_Ad_ID']
#         mcf_uuid = hashlib.md5(mcf_job_ad_id.encode()).hexdigest()

#         print(f">> Randomised MCF job ad UUID: {mcf_uuid}")

#     # Otherwise, assume user wants to generate prediction for a new job ad
#     elif (mcf_url[0:4] == "MCF-") or (mcf_url[0:4] == "JOB-"):

#         print(">> User wants to generate prediction using the MCF job ad ID")

#         # Assume the URL parameter contains the job ad ID and hash it
#         mcf_uuid = hashlib.md5(mcf_url.encode()).hexdigest()
#         print(f'>> Queried MCF job ad UUID: {mcf_uuid}')

#     else:

#         print(">> User wants to generate prediction using the MCF job ad URL")

#         # Note that we append a "?" to the end of the URL to ensure we capture the ID correctly
#         regex_matches = re.search('\\-{1}([a-z0-9]{32})\\?', mcf_url + "?")

#         # Try extracting the MCF job ad ID from the URL
#         try:
#             mcf_uuid = regex_matches.group(1)
#             print(f'>> Queried MCF job ad UUID: {mcf_uuid}')
#         except:
#             print('>> Could not identify hashed MCF job ad ID from the URL.')
#             raise HTTPException(status_code = 404, 
#                                 detail = "Invalid URL from MyCareersFuture. Please check that you have a valid URL and call this API again.")

#     return mcf_uuid