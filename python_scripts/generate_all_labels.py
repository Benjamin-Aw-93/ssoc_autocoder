import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} {levelname:<8} {message}",
    style='{',
    filename="%slog" % __file__[:-2],
    filemode='w'
)

def extract_labelled_excel(data):
    """
        Extract the rows where by `Label As SSOC` is 'y'.

        Args:
            data - excel files.
        
        Returns:
            the filtered excel file.
    """
    try:
        data["Label As SSOC"] = data["Label As SSOC"].str.strip()
        data["Label As SSOC"] = data["Label As SSOC"].str.lower()
        return data[data["Label As SSOC"] == 'y']
    except AttributeError:
        return []

input_json = sys.argv[1]
with open(input_json, "r") as f:
    config = json.load(f)

inputs = config['generate_all_labels']
folder_path = inputs['folder_path']
excel_folder_path = inputs['excel_folder_path']

# stores {uuid: ssoc} pairs
uuid_ssoc_dict = {}

# stores the conflict in the follow format
# {'uuid1': [ssoc1, ssoc2], 'uuid2': [ssoc3, ssoc4, ssoc5]}
conflict = {}

for file in os.listdir(excel_folder_path):
    file_path = os.path.join(excel_folder_path, file)
    excel = pd.read_excel(file_path) 
    
    # extract the ssoc from the excel name
    ssoc = file[:5]
    logging.info(f"Scanning for {ssoc}")
    excel = extract_labelled_excel(excel)
    if len(excel) == 0:
        logging.info(f"{ssoc} does not have any verified ads, going to the next available ssoc")
        continue

    # potential uuids
    candidate_uuid = excel['uuid']

    # do a check for every potential uuid
    for uuid in candidate_uuid: 
        if uuid in uuid_ssoc_dict: # key already present, add into conflict
            if uuid in conflict: # there is already a key: value(list) created in the conflict dictionary
                conflict[uuid].append(ssoc)
            else: # uuid does not exist as a key in the conflict dictionary yet, got to create a new {key: value} pair
                conflict[uuid] = [uuid_ssoc_dict[uuid], ssoc]

        else: # {uuid: ssoc} does not exist yet, add into the dictionary
            uuid_ssoc_dict[uuid] = ssoc

if len(conflict) > 0: # if conflict is not empty, let the user know what the conflicts are
    logging.info(f"There are {len(conflict)} uuid conflicts, please resolve them before generating the `all_labels.json` file.")

    # generate conflict
    for conflicted_uuid in conflict:
        logging.info(f"Conflicted uuid '{conflicted_uuid}' appears in SSOCs {conflict[conflicted_uuid]}.")

else: # there are no conflicts, generate all_labels.json
    logging.info("There are no conflicts in uuids! Creating all_labels.json file...")
    all_labels = {}

    # extract the timestamp
    timestamp = datetime.now()
    str_date_time = timestamp.strftime("%d-%m-%Y-%H-%M-%S")
    
    for file in os.listdir(excel_folder_path):
        file_path = os.path.join(excel_folder_path, file)
        excel = pd.read_excel(file_path)
        excel = extract_labelled_excel(excel)
        
        if len(excel) == 0:
            continue

        logging.info(f"In {file}, there are {len(excel)} uuids added into all_labels.json file")

        # extract the ssoc from the excel name
        ssoc = file[:5]

        for index, row in excel.iterrows(): # iterate every row in the dataframe
            all_labels[row['jobPostId']] = {"ym": row['yearMonth'], "ssoc": ssoc} # add into the all_labels dictionary

    if len(all_labels) == 0:
        logging.info(f"There are no verified job ads at all. Please verify them in order to create any all_labels.json")
    
    else:
        # serializing json
        json_object = json.dumps(all_labels, indent=4)

        # writing to the json file
        with open(f"{folder_path}/all_labels_{str_date_time}.json", "w") as f:
            f.write(json_object)
        f.close()

        logging.info(f"all_labels_{str_date_time}.json has been created!")