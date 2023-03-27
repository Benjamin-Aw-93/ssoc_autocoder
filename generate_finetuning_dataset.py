import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} {levelname:<8} {message}",
    style='{',
    filename="%slog" % __file__[:-2],
    filemode='w'
)

directory_path = "data/labels/cosine_similarity"

directory = os.listdir(directory_path)
latest_timestamp = directory[0][11:30]

# locate the json file with the latest timestamp, [11:30] is the index of the timestamp in the file path
for file in directory:
    if latest_timestamp < file[11:30]:
        latest_timestamp = file[11:30]

f = open(f"data/labels/cosine_similarity/all_labels_{latest_timestamp}.json")
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

# extract the timestamp
timestamp = datetime.now()
str_date_time = timestamp.strftime("%d-%m-%Y-%H-%M-%S")

# storing the finetuning dataset
df.reset_index(inplace=True, drop=True)
df.to_csv(f"data/finetuning_dataset/finetuning_data_{latest_timestamp}", index=False)