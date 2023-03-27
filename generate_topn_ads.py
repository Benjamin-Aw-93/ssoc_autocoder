# import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
import datetime
import pathlib2 as pl2
import json
import logging

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

excel_folder_path = "data/labels/cosine_similarity/excel"

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
    data[['uuid', 'jobPostId', 'title', 'cleaned_description', 'ssocCode', 'cosine_similarity', 'Label as SSOC']].to_excel(file_path)
    logging.info(f"Data has been exported to {file_path}")

def generate_top_n_closest(ssoc_input, detailed_def, detailed_def_embeddings, data, embeddings, 
                            n=10, threshold=0, check_duplicates=False):
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
    print(f"Generating top {n} closest ads for {ssoc_input}")
    # get the ssoc representative vector using the detailed definition
    ssoc_representative_index = detailed_def[detailed_def['SSOC 2020'] == ssoc_input].index
    ssoc_representative = detailed_def_embeddings[ssoc_representative_index]

    if check_duplicates == True: # want to check duplicates, remove the duplicated indexes from the embeddings
        cur_excel = pd.read_excel(f"{excel_folder_path}/{ssocInput}.xlsx")

        # extract the existing job IDs
        existing_jobIDs = cur_excel['jobPostId'] 
        
        # extract the indices for the duplicated job IDs and remove from the embeddings array
        merged = data.merge(existing_jobIDs, how='left', on='jobPostId', indicator=True)
        index_to_remove = merged[merged['_merge'] == 'both'].index

        embeddings = np.delete(embeddings, index_to_remove, axis=0)

    # Perform cosine similarity
    cosine_sim = cosine_similarity(embeddings, ssoc_representative)

    # reshape as the output of cosine_similarity is [[cosine sim 1], [cosine sim 2],..., [cosine sim n]]
    cosine_sim = cosine_sim.reshape(cosine_sim.shape[0])

    # np.argsort returns the index of ascending sorted array
    # reverse the array by [::-1]
    # get the top 10 by [:10]
    topn_index = np.argsort(cosine_sim)[::-1][:n]

    # locate the top n job ads based on the cosine similarity
    topn_ads = data.loc[topn_index]
    topn_ads['cosine_similarity'] = cosine_sim[topn_index]
    topn_ads = topn_ads[topn_ads['cosine_similarity'] >= threshold]
    
    # print(f"Top {n} closest ads generated", end="\n\n")
    return topn_ads

yearmonth = "2022-07"
n = 10
rareList = ["25291", "41320"]

detailed_def_embeddings_filepath = "data/detailed_def_embeddings.pickle"
data_filepath = f"data/processed/{yearmonth}/{yearmonth}.csv" # year-month can be input
data_embeddings_filepath = f"data/processed/{yearmonth}/embeddings.pickle"

# read embeddings and data
with open(detailed_def_embeddings_filepath, "rb") as f: 
    detailed_def_embeddings = pickle.load(f)

with open(data_embeddings_filepath, "rb") as f:
    data_embeddings = pickle.load(f)

data = pd.read_csv(data_filepath)
detailed_def = pd.read_excel("SSOC2020 Detailed Definitions.xlsx", header=4)

for ssocInput in rareList:
    logging.info(f"Generating top{n} for {ssocInput}")
    if check_existing_file(ssocInput, excel_folder_path): # excel exist, check duplicates and append
        logging.info(f"Excel file already exist, checking for duplicates and appending into the file")
        cur_excel = pd.read_excel(f"{excel_folder_path}/{ssocInput}.xlsx")

        topn_data = generate_top_n_closest(ssocInput, detailed_def, detailed_def_embeddings, data, data_embeddings, check_duplicates=True)
        topn_data['Label as SSOC'] = ''

        # append 
        to_export = pd.concat([cur_excel, topn_data], ignore_index=True)
        # to_export = cur_excel.append(topn_data)
        export_topn_excel(f"{excel_folder_path}/{ssocInput}.xlsx", to_export)
        
    else: # excel does not exist, create it
        logging.info(f"Creating excel file for {ssocInput}")
        topn_data = generate_top_n_closest(ssocInput, detailed_def, detailed_def_embeddings, data, data_embeddings)
        topn_data['Label as SSOC'] = ''
        export_topn_excel(f"{excel_folder_path}/{ssocInput}.xlsx", topn_data)
    

    






