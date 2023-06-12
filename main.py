import logging
from ssoc_autocoder.deployment.logging import addLoggingLevel
logging.basicConfig(format = "%(asctime)s | autocoder-logs | %(levelname)s | %(message)s", datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger("autocoder")
logger.setLevel(logging.INFO)
addLoggingLevel("QUERY", 25, methodName=None)

# Importing the required libraries
import json
import pickle
import random
import time

# Importing our custom scripts
import ssoc_autocoder.deployment.api as api

# Initialising the FastAPI object
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Python typing
from typing import Optional, List
from enum import Enum

# Initialising the important artifacts
model, tokenizer, encoding, ssoc_desc, job_title_map = api.initialise()

app_description = """
SSOC Autocoder API returns to you the corresponding SSOC codes based on job descriptions. 

## What is the SSOC Autocoder about?

With this API, you will be able to:

**Query SSOC** (_not implemented_)
"""
tags_metadata = [
    {"name": "Base",
     "description": "This API is to check if the API has been loaded or not. Can be removed once deployed."},
    {"name": "SSOC Prediction",
     "description": "This API call serves to return the top 10 most appropriate SSOC Codes based on the corresponding job descriptions. Users will have to decide if they want to query from an existing entry on MCF; or enter the job description as free text."},
]

app = FastAPI(
    title="SSOC Autocoder",
    description = app_description,
    version="0.1.0",
    contact = {
        "name": "Shaun Khoo",
        "email": "shaun@dsaid.gov.sg",
    },
    openapi_tags = tags_metadata,
    # docs_url = None,
    # redoc_url = "/documetation"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags = ["Base"])
def health_check():
    return "SSOC Autocoder API is active."
    
class TypeName(str, Enum):
    id = "id"
    text = "text"
    
@app.get('/prediction', tags = ["SSOC Prediction"])
def prediction(query_type: str = Query(title = "Query Type", 
                                       description = "Choose `text` to pass in the job title and job description, `id` to pass in the MyCareersFuture job ad ID, `url` to pass in the MyCareersFuture job ad URL, or `feelinglucky` to use a random job ad."), 
               job_title: str = Query(title = "Job Title",
                                      default = None,
                                      description = "Enter the job title from the job ad. Only works if `query_type` is `text`."),
               job_desc: str = Query(title = "Job Description",
                                     default = None,
                                     description = "Enter the job description from the job ad, preferably with the HTML intact. Only works if `query_type` is `text`."),
               id: str = Query(title = "Job Ad ID",
                               default = None,
                               description = "**MyCareersFuture only** Enter the job ad ID from the MyCareersFuture job ad. Only works if `query_type` is `id`."),
               url: str = Query(title = "Job Ad URL",
                                default = None,
                                description = "**MyCareersFuture only** Enter the URL to the MyCareersFuture job ad. Only works if `query_type` is `url`."),               
               n_results: int = Query(title = "Enter the number of results to be returned. Defaulted to `10` entries",
                                      default = 10,
                                      description = "Returns the top n results from the model. Defaulted value is `10`"),
               return_occupation: bool = Query(title = "Return a list of WPD Job title list based on each predicted SSOC value.",
                                        default = False,
                                        description = "Retun list of Job titles based on WPD's SSOC to job title mapping. Default value is `False`"),
               return_confidence_score: bool = Query(title = "Returns the confidence score in the final output.",
                                                     default = False,
                                                     description = "Returns the model's confidences of each given prediction. Default value is `False`"),
               return_description: bool = Query(title = "Returns the SSOC description of each predicted SSOC value",
                                                default = False,
                                                description = "Returns the given SSOC description provided by DOS. Default value is `False`") 
              ):

    # Timer for the overall prediction API
    start_time = time.time()

    logger.query(f"{query_type} | {job_title} | {job_desc} | {id} | {url} | {n_results} | {return_occupation} | {return_confidence_score} | {return_description}")

    # Validate parameters
    api.validate_parameters(query_type, job_title, job_desc, id, url, n_results)

    # Start timing how long the querying of the MCF jobs API takes
    jobs_api_start = time.time()

    # For the text input type, we can directly query the model using the text
    if query_type == "text":
        job_id = None
    
    # For all other input types, we need to ping the MCF API first
    else:

        # If the user is passing in the job ad ID
        if query_type == "id":

            # Call the function to convert the job ad ID to the UUID
            mcf_uuid = api.convert_to_uuid(id, input_type = "id")

        # If the user is passing in the MCF job ad URL
        elif query_type == "url":

            # Call the function to extract the data from the MCF URL
            mcf_uuid = api.convert_to_uuid(url, input_type = "url")

        # If the user wants a random job ad
        elif query_type == "feelinglucky":
            mcf_uuid = api.convert_to_uuid(None, input_type = "feelinglucky")

        job_id, job_title, job_desc = api.call_mcf_api(mcf_uuid)

    logger.info(f'TIMING | MCF job API call: {(time.time() - jobs_api_start):.2f}s')

    # Start timing how long the querying of the model takes
    model_prediction_start = time.time()

    # Call the model using the job title and job description
    output = api.generate_predictions(model, tokenizer, encoding, ssoc_desc, job_title, job_desc, job_title_map, n_results, return_occupation, return_confidence_score, return_description, job_id)
    logger.info(f'TIMING | Model prediction: {(time.time() - model_prediction_start):.2f}s')

    logger.info(f'TIMING | Overall: {(time.time() - start_time):.2f}s')

    return output

class PredictionQuery(BaseModel):
    query_type: str = Field(title = "Query Type", 
                            description = "Choose `text` to pass in the job title and job description, `id` to pass in the MyCareersFuture job ad ID, `url` to pass in the MyCareersFuture job ad URL, or `feelinglucky` to use a random job ad.")
    job_title: str = Field(title = "Job Title",
                           default = None,
                           description = "Enter the job title from the job ad. Only works if `query_type` is `text`.")
    job_desc: str = Field(title = "Job Description",
                          default = None,
                          description = "Enter the job description from the job ad, preferably with the HTML intact. Only works if `query_type` is `text`.")
    id: str = Field(title = "Job Ad ID",
                    default = None,
                    description = "**MyCareersFuture only** Enter the job ad ID from the MyCareersFuture job ad. Only works if `query_type` is `id`.")
    url: str = Field(title = "Job Ad URL",
                                default = None,
                                description = "**MyCareersFuture only** Enter the URL to the MyCareersFuture job ad. Only works if `query_type` is `url`.")
    n_results: int = Field(title = "Number of predictions",
                           default = 10,
                           description = "Returns the top n results from the model. Defaulted value is `10`")
    return_occupation: bool = Field(title = "Return a list of WPD Job title list based on each predicted SSOC value.",
                                    default = False,
                                    description = "Retun list of Job titles based on WPD's SSOC to job title mapping. Default value is `False`")
    return_confidence_score: bool = Field(title = "Returns the confidence score in the final output.",
                                          default = False,
                                          description = "Returns the model's confidences of each given prediction. Default value is `False`")
    return_description: bool = Field(title = "Returns the SSOC description of each predicted SSOC value",
                                     default = False,
                                     description = "Returns the given SSOC description provided by DOS. Default value is `False`")

    
@app.post('/prediction', tags = ["SSOC Prediction"])
def prediction(query: PredictionQuery):

    # Timer for the overall prediction API
    start_time = time.time()

    logger.query(f"{query.query_type} | {query.job_title} | {query.job_desc} | {query.id} | {query.url} | {query.n_results} | {query.return_occupation} | {query.return_confidence_score} | {query.return_description}")

    # Validate parameters
    api.validate_parameters(query.query_type, query.job_title, query.job_desc, query.id, query.url, query.n_results)

    # Start timing how long the querying of the MCF jobs API takes
    jobs_api_start = time.time()

    # For the text input type, we can directly query the model using the text
    if query.query_type == "text":
        job_id = None
    
    # For all other input types, we need to ping the MCF API first
    else:

        # If the user is passing in the job ad ID
        if query.query_type == "id":

            # Call the function to convert the job ad ID to the UUID
            mcf_uuid = api.convert_to_uuid(query.id, input_type = "id")

        # If the user is passing in the MCF job ad URL
        elif query.query_type == "url":

            # Call the function to extract the data from the MCF URL
            mcf_uuid = api.convert_to_uuid(query.url, input_type = "url")

        # If the user wants a random job ad
        elif query.query_type == "feelinglucky":
            mcf_uuid = api.convert_to_uuid(None, input_type = "feelinglucky")

        query.job_id, query.job_title, query.job_desc = api.call_mcf_api(mcf_uuid)

    logger.info(f'TIMING | MCF job API call: {(time.time() - jobs_api_start):.2f}s')

    # Start timing how long the querying of the model takes
    model_prediction_start = time.time()

    # Call the model using the job title and job description
    print(query.n_results)
    print(dir(query.n_results))
    output = api.generate_predictions(model, tokenizer, encoding, ssoc_desc, query.job_title, query.job_desc, job_title_map, query.n_results, query.return_occupation, query.return_confidence_score, query.return_description, job_id)
    logger.info(f'TIMING | Model prediction: {(time.time() - model_prediction_start):.2f}s')

    logger.info(f'TIMING | Overall: {(time.time() - start_time):.2f}s')

    return output

@app.get('/embeddings', tags = ["Autocoder Embeddings"])
def prediction(id: str = Query(title = "Job Ad ID"  ,
                               default = None,
                               description = "Enter the job ad ID from the MyCareersFuture job ad. Only works if `query_type` is `id`.")):
    
    # Timer for the overall prediction API
    start_time = time.time()

    logger.query(f"{id}")
    
    # Call the function to convert the job ad ID to the UUID
    mcf_uuid = api.convert_to_uuid(id, input_type = "id")

    job_id, job_title, job_desc = api.call_mcf_api(mcf_uuid)

    logger.info(f'TIMING | MCF job API call: {(time.time() - start_time):.2f}s')

    # Start timing how long the querying of the model takes
    model_prediction_start = time.time()

    # Call the model using the job title and job description
    output = api.generate_embeddings(model, tokenizer, job_title, job_desc)
    
    logger.info(f'TIMING | Model prediction: {(time.time() - model_prediction_start):.2f}s')

    logger.info(f'TIMING | Overall: {(time.time() - start_time):.2f}ss')

    return output