# Importing required libraries
import json
import random
import hashlib
import re
import pickle

# Importing the deep learning libraries
from transformers import DistilBertTokenizer
import torch

# Importing our custom scripts
import model_prediction
from ssoc_autocoder import model_training
from ssoc_autocoder import processing

# Initialising the HTTP object for the API call
import urllib3
http = urllib3.PoolManager()

# Initialising the FastAPI object
from fastapi import FastAPI, HTTPException
app = FastAPI()

# Function for initialise the required objects for prediction
def initialise():

    model_filepath = 'artifacts/ssoc-autocoder-model.pickle'
    tokenizer_filepath = 'artifacts/distilbert-tokenizer-pretrained-7epoch'
    ssoc_idx_encoding_filepath = 'artifacts/ssoc-idx-encoding.json'

    # Load the model and tokenizer objects
    with open(model_filepath, 'rb') as handle:
        model = pickle.load(handle)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_filepath)

    # Reading in the SSOC-index encoding
    encoding = model_training.import_ssoc_idx_encoding(ssoc_idx_encoding_filepath)

    return model, tokenizer, encoding

def return_uuid(mcf_url: str):
    """
    Extracts the MCF UUID from the provided URL.
    """

    print("> Generating ")

    # Check if the user wants to randomly grab a previous query
    if mcf_url == 'feelinglucky':
        
        print(">> User wants a random previous query")

        # Reading in the dummy data
        with open('artifacts/dummy_data.json') as json_file:
            dummy_data = json.load(json_file)

        # Select one of the dummy data randomly
        data = dummy_data[random.randint(0,49)]

        # Retrieve the MCF job ad ID and overwriting it with one of our samples
        mcf_job_ad_id = data['MCF_Job_Ad_ID']
        mcf_uuid = hashlib.md5(mcf_job_ad_id.encode()).hexdigest()

        print(f">> Randomised MCF job ad UUID: {mcf_uuid}")

    # Otherwise, assume user wants to generate prediction for a new job ad
    else:

        print(">> User wants to generate prediction for a new MCF job ad")

        # Note that we append a "?" to the end of the URL to ensure we capture the ID correctly
        regex_matches = re.search('\\-{1}([a-z0-9]{32})\\?', mcf_url + "?")

        # Try extracting the MCF job ad ID from the URL
        try:
            mcf_uuid = regex_matches.group(1)
            print(f'>> Queried MCF job ad UUID: {mcf_uuid}')
        except:
            print('>> Could not identify hashed MCF job ad ID from the URL.')
            raise HTTPException(status_code = 404, 
                                detail = "Invalid URL from MyCareersFuture. Please check that you have a valid URL and call this API again.")

    return mcf_uuid

def call_mcf_api(mcf_uuid):
    """
    Calls the MCF jobs API with the provided UUID, with
    in-built error handling.
    """

    # Call the MCF API
    resp = http.request('GET', f'https://api.mycareersfuture.gov.sg/v2/jobs/{mcf_uuid}')

    # Error handling if an invalid MCF job ad ID is provided
    if resp.status != 200 or json.loads(resp.data) == {'message': 'UUID is not found in the database.'}:
        raise HTTPException(status_code = 404, 
                    detail = "Error calling the MCF Jobs API with the provided MCF URL. Please check that you have a valid MCF URL and call this API again.")
 
    print('MCF API called successfully')
    
    return resp

def generate_predictions(model, tokenizer, encoding, mcf_url, resp, ssoc_desc):
    """
    Generates and formats the predictions for output.
    """

    # If not, we extract the job title and description
    mcf_job_ad_id = json.loads(resp.data)['metadata']['jobPostId']
    mcf_job_title = json.loads(resp.data)['title']
    mcf_job_desc = json.loads(resp.data)['description']

    # Generate predictions
    if mcf_url == 'feelinglucky':

        with open('artifacts/dummy_data.json') as json_file:
            dummy_data = json.load(json_file)
        data = [entry for entry in dummy_data if (entry['MCF_Job_Ad_ID'] == mcf_job_ad_id)][0]

        # Appending the SSOC title and description to the output
        for prediction in data['predictions']:
            prediction['SSOC_Title'] = ssoc_desc[prediction['SSOC_Code']]['title']
            prediction['SSOC_Description'] = ssoc_desc[prediction['SSOC_Code']]['description']

        # Assigning it to the final predictions object
        predictions_formatted = data['predictions']

        print('Formatted previous query correctly for output')

    else:

        print('Loading custom neural network and generating prediction')
        predictions = model_prediction.generate_single_prediction(model, 
                                                                  tokenizer,
                                                                  mcf_job_title,
                                                                  processing.process_text(mcf_job_desc),
                                                                  None,
                                                                  encoding)['SSOC_5D']

        predictions_formatted = []
        for i in range(0, 10):
            prediction_formatted = {
                'SSOC_Code': str(predictions['predicted_ssoc'][i]),
                'SSOC_Title': ssoc_desc[str(predictions['predicted_ssoc'][i])]['title'],
                'SSOC_Description': ssoc_desc[str(predictions['predicted_ssoc'][i])]['description'],
                'Prediction_Confidence': f"{predictions['predicted_proba'][i]*100:.2f}%"
            }
            predictions_formatted.append(prediction_formatted)

        print('Formatted predictions correctly for output')

    # Initialise output with the MCF information
    output = {
        'mcf_job_id': mcf_job_ad_id,
        'mcf_job_title': mcf_job_title,
        'mcf_job_desc': mcf_job_desc,
        'top_prediction': predictions_formatted[0],
        'other_predictions': predictions_formatted[1:10]
    }

    return output

print("Initialising model, tokenizer, and encoding objects...\r", end = "")
model, tokenizer, encoding = initialise()
with open('artifacts/ssoc_desc.json') as json_file:
    ssoc_desc = json.load(json_file)
print("Initialising model, tokenizer, and encoding objects... complete!")

@app.get("/")
def read_root():
    return "SSOC Autocoder API is active and ready."

@app.get('/predict')
def get_prediction(mcf_url: str):        

    # Extract the correct MCF job ad UUID
    mcf_uuid = return_uuid(mcf_url)
    resp = call_mcf_api(mcf_uuid)

    # Generate the predictions
    output = generate_predictions(model, tokenizer, encoding, mcf_url, resp, ssoc_desc)

    print('Script completed successfully!')

    return output

