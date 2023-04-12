import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} {levelname:<8} {message}",
    style='{',
    filename="%slog" % __file__[:-2],
    filemode='w'
)

input_json = sys.argv[1]
with open(input_json, "r") as f:
    config = json.load(f)

inputs = config["generate_finetuning_dataset"]
directory_path = inputs["directory_path"]
finetuning_dir_path = inputs["finetuning_dir_path"]
test_size = inputs['test_size']

directory = os.listdir(directory_path)
latest_timestamp = directory[0][11:30]
latest_timestamp_datetime = datetime.strptime(latest_timestamp, "%d-%m-%Y-%H-%M-%S")

# locate the json file with the latest timestamp, [11:30] is the index of the timestamp in the file path
for file in directory:
    if file == "excel": # does not have timestamp, ignore
        continue
    cur_timestamp = file[11:30]
    cur_timestamp_datetime = datetime.strptime(file[11:30], "%d-%m-%Y-%H-%M-%S")
    if latest_timestamp_datetime < cur_timestamp_datetime:
        latest_timestamp = file[11:30]
f = open(f"{directory_path}/all_labels_{latest_timestamp}.json")
dictionary = json.load(f)
logging.info(f"There are {len(dictionary)} job IDs to be corrected")

# store the final df to be converted to CSV
df = pd.DataFrame()

for key in dictionary:
    # separating the components of the key: value pair
    job_id = key
    ym = dictionary[key]['ym']
    ssoc = dictionary[key]['ssoc']

    # read the year-month data
    ym_data = pd.read_csv(f"data/processed/{ym}/{ym}.csv")

    # locate the data that belongs to the corresponding job-id
    data = ym_data[ym_data['jobPostId'] == job_id]
    data = data[['jobPostId', 'title', 'cleaned_description']]
    data['ssocCode'] = ssoc

    df = pd.concat([df, data])

if len(df) <= 1:
    logging.info(f"There are only {len(df)} number of datapoints. Please ensure you have more than 1 in order to create a train-test split.")
else:
    # create directory
    try:
        os.mkdir(f"{finetuning_dir_path}/{latest_timestamp}")
    except FileExistsError: # directory already exist
        pass
    logging.info(f"Creating a train-test split of test size = {test_size}")
    # train test split
    train, test = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)

    # storing the train, test dataset
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    train.to_csv(f"{finetuning_dir_path}/{latest_timestamp}/train.csv", index=False)
    test.to_csv(f"{finetuning_dir_path}/{latest_timestamp}/test.csv", index=False)