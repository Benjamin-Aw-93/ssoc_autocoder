# import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
import datetime
import pathlib2 as pl2
import json
import logging
import sys

import os
import os.path
from os import path

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} {levelname:<8} {message}",
    style='{',
    filename="%slog" % __file__[:-2],
    filemode='w'
)

def check_existing_file(ssoc_input, folder_path):
    """
        Checks whether we have already generated and saved the top-n closest ads for a particular SSOC.
        
        Args:
            ssoc_input: the SSOC that we want to know if there already exist the file in the folder
            folder_path: the folder that stores all the top-n closest ads to a SSOC 

        Returns:
            True: there already exist a file for the SSOC
            False: there does not exist a file for the SSOC
    """

    files = os.listdir(folder_path)
    for file in files:
        if str(file).find(ssoc_input) != -1: # excel file already exist
            return True
    return False

def export_topn_excel(file_path, data):
    """
        Exports a dataframe to an excel file

        Args:
            file_path: the path where we want to store the dataframe in
            data: the data that we want to store
    """
    data[['uuid', 'jobPostId', 'title', 'cleaned_description', 'ssocCode', 'cosine_similarity', 'yearMonth', 'Label As SSOC']].to_excel(file_path)
    logging.info(f"Data has been exported to {file_path}")

def generate_top_n_closest(ssoc_input, detailed_def, detailed_def_embeddings, data, embeddings, 
                            n, threshold=0, check_duplicates=False):
    """
        Generates the top-n closest ads to a SSOC based on cosine similarity between the detailed definitions' embeddings and the cleaned descriptions' embeddings.

        Args:
            ssoc_input: ssoc that we want to find the top-n job ads for.
            detailed_def: excel sheet of the detailed definitions.
            detailed_def_embeddings: embeddings of the detailed defintiions.
            data: the data we are trying to extract the top 10 closest ads from.
            embeddings: the embeddings of the descriptions in the data.
            n: top-n job ads, default value is 10.
            threshold: cosine similarity threshold, default value is 0.

        Returns:
            A dataframe with the top-n closest ads.
    """
    # get the ssoc representative vector using the detailed definition
    ssoc_representative_index = detailed_def[detailed_def['SSOC 2020'] == ssoc_input].index
    ssoc_representative = detailed_def_embeddings[ssoc_representative_index]

    # Perform cosine similarity
    cosine_sim = cosine_similarity(embeddings, ssoc_representative)

    # reshape as the output of cosine_similarity is [[cosine sim 1], [cosine sim 2],..., [cosine sim n]]
    cosine_sim = cosine_sim.reshape(cosine_sim.shape[0])

    # sort and retrieve the index
    # np.argsort returns the index of ascending sorted array
    # reverse the array by [::-1]
    sorted_cosine_sim_index = np.argsort(cosine_sim)[::-1]

    if check_duplicates == True: # want to check duplicates and ensure we don't append them into the current excel again
        cur_excel = pd.read_excel(f"{excel_folder_path}/{ssocInput}.xlsx")

        # extract the existing job IDs
        existing_jobIDs = cur_excel['jobPostId'] 
        
        # extract the indices for the duplicated job IDs and remove from the embeddings array
        merged = data.merge(existing_jobIDs, how='left', on='jobPostId', indicator=True)

        # if the jobID exist in both dataframes, '_merge' will be marked as 'both' value
        # extract the index of the common datapoints so that we can remove them from the sorted_cosine_sim_index array
        index_to_remove = merged[merged['_merge'] == 'both'].index

        # the set difference between the index of the sorted cosine similarity and the index to remove
        sorted_cosine_sim_index = np.setdiff1d(sorted_cosine_sim_index, index_to_remove, assume_unique=True)

    # get the top 10 by [:10]
    topn_index = sorted_cosine_sim_index[:n]

    # locate the top n job ads based on the cosine similarity
    topn_ads = data.loc[topn_index]
    topn_ads['cosine_similarity'] = cosine_sim[topn_index]
    topn_ads = topn_ads[topn_ads['cosine_similarity'] >= threshold]

    logging.info(f"Top {len(topn_ads)} ads for {ssoc_input} has been generated")

    return topn_ads

# read the inputs
input_json = sys.argv[1]
with open(input_json, "r") as f:
    config = json.load(f)

inputs = config["generate_topn_ads"]
excel_folder_path = inputs["excel_folder_path"]
year_month = inputs["year_month"] 
n = inputs["n"] 
rare_list = inputs["rare_list"] 

detailed_def_embeddings_filepath = inputs['detailed_def_embeddings_filepath']
detailed_def_filepath = inputs['detailed_def_filepath']
data_filepath = f"data/processed/{year_month}/{year_month}.csv" 
data_embeddings_filepath = f"data/processed/{year_month}/embeddings.pickle"

# read embeddings and data
with open(detailed_def_embeddings_filepath, "rb") as f: 
    detailed_def_embeddings = pickle.load(f)

with open(data_embeddings_filepath, "rb") as f:
    data_embeddings = pickle.load(f)

data = pd.read_csv(data_filepath)
detailed_def = pd.read_excel(detailed_def_filepath, header=4)

for ssocInput in rare_list:
    if ssocInput not in detailed_def['SSOC 2020'].values: # invalid SSOC, move on to the next one
        logging.info(f"SSOC {ssocInput} is invalid, going to the next SSOC")
        continue

    logging.info(f"Generating top {n} for {ssocInput}")
    if check_existing_file(ssocInput, excel_folder_path): # excel exist, check duplicates and append
        logging.info(f"Excel file already exist, checking for duplicates and appending into the file")
        cur_excel = pd.read_excel(f"{excel_folder_path}/{ssocInput}.xlsx")

        topn_data = generate_top_n_closest(ssocInput, detailed_def, detailed_def_embeddings, data, data_embeddings, n, check_duplicates=True)
        topn_data['Label As SSOC'] = ''
        topn_data['yearMonth'] = year_month

        # append 
        to_export = pd.concat([cur_excel, topn_data], ignore_index=True)
        export_topn_excel(f"{excel_folder_path}/{ssocInput}.xlsx", to_export)
        logging.info(f"Excel for {ssocInput} has been exported to labels/cosine_similarity")
        
    else: # excel does not exist, create it
        logging.info(f"Creating excel file for {ssocInput}")
        topn_data = generate_top_n_closest(ssocInput, detailed_def, detailed_def_embeddings, data, data_embeddings, n)
        topn_data['Label As SSOC'] = ''
        topn_data['yearMonth'] = year_month
        export_topn_excel(f"{excel_folder_path}/{ssocInput}.xlsx", topn_data)
        logging.info(f"Excel for {ssocInput} has been exported to labels/cosine_similarity")