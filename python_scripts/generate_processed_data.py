import pandas as pd
import logging
import sys
import json
import os
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm.notebook import tqdm, trange
import torch
import pickle
import numpy as np
import text_cleaning
import datetime
from dateutil.rrule import rrule, MONTHLY

"""
    This script 

"""

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} {levelname:<8} {message}",
    style='{',
    filename="%slog" % __file__[:-2],
    filemode='w'
)

def extract_mcf_data(json):
    """
        Extracts the necessary data from the json file and convert them into a dictionary.

        Args:
            json - contains the data.

        Returns:
            output - dictionary version of the necessary data.
            output_date - posting date of the data.

    """

    output = {}
    transfer = ['uuid', 'title', 'description', 'minimumYearsExperience', 'numberOfVacancies', 'ssocCode']
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
    metadata = ['jobPostId', 'originalPostingDate', 'newPostingDate', 'expiryDate', 'totalNumberOfView', 'totalNumberJobApplication']
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

# convert mcf ssoc to current ssoc
def mcfssoc_to_ssoc2020(mcf_ssoc_converter, data):
    """
        Converts SSOC from MCF SSOC to SSOC 2020.

        Args:
            mcf_ssoc_converter: the mapping from MCF SSOC to SSOC 2020.
            data: the data we want to convert the SSOC for.

        Returns:
            data: consist of a new column `ssocCode` which is the SSOC 2020.
    """
    # SsocCode is MCF SSOC
    mcf_ssoc_converter.rename(columns={'SsocCode': 'mcf_ssoc_code'}, inplace=True)
    data.rename(columns={'ssocCode': 'mcf_ssoc_code'}, inplace=True)

    mcf_ssoc_converter = mcf_ssoc_converter[['mcf_ssoc_code', 'ActualSsocCode']]

    # merge based on mcf_ssoc_code
    data = data.merge(mcf_ssoc_converter, how='left', on='mcf_ssoc_code')

    # ActualSsocCode is SSOC2020
    data.rename(columns={'ActualSsocCode': 'ssocCode'}, inplace=True)

    return data

# generate cleaned description
def generate_cleaned_description(data):
    """
        Cleans the `description` column and create a new column `cleaned_description`.

        Args:
            data - the data that we want to clean.
        
        Returns:
            data - the dataframe that consist of a new column `cleaned_description`.
    """
    cleaned_desc = []
    data['description'] = list(data['description'])
    for desc in tqdm(data['description']):
        try:
            cleaned_desc.append(text_cleaning.clean_html_unicode(text_cleaning.clean_raw_string(desc)))
        except:
            print(desc)

    data['cleaned_description'] = cleaned_desc
    print("Descriptions cleaned")

    return data

# generate embeddings
def generate_embeddings(model, tokenizer, data): # returns embeddings and the new dataframe with the column 'cleaned_description'
    """
        Generates the embeddings for cleaned description using the LLM model.

        Args:
            model: the model that generates the embeddings.
            tokenizer: the tokenizer to tokenize `cleaned_description`.
            data: data that we want to generate the embeddings from.
        
        Returns:
            embeddings: the embeddings generated.
    """
    logging.info("Tokenizing and generating embeddings...")
    batch_size = 64
    embeddings = []
    logging.info(f"There are {len(data)/batch_size} batches of 64s")

    # Split up the entire dataset into batches
    for i in trange(0, len(data), batch_size):
        
        # For each batch, tokenize the text
        text = data['cleaned_description'][i:i+64].tolist()
        title = data['title'].tolist()
        tokenized = tokenizer(
                text=text,
                text_pair=None,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True)

        tokenized_title = tokenizer(
                text=title,
                text_pair=None,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True)

        # Extract the title IDs and masks
        title_ids = torch.tensor(tokenized_title['input_ids'], dtype=torch.long)
        title_masks = torch.tensor(tokenized_title['attention_mask'], dtype=torch.long)

        # Extract the IDs and masks
        text_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
        text_masks = torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        
        # Run the data through the model and extract the embeddings tensor
        with torch.no_grad():
            text_embeddings = model.l1(text_ids, text_masks) 
            title_embeddings = model.l1(title_ids, title_masks)
            title_hidden_state = title_embeddings[0]
            title_vec = title_hidden_state[:, 0]

            text_hidden_state = text_embeddings[0]
            text_vec = text_hidden_state[:, 0]
            
            embeddings.extend(text_vec.numpy().tolist())
        #     # embeddings_tensor = preds[0][:,0]
        #     embeddings_tensor = torch.tensor(1)
            
        # # Append it to the output list
        # embeddings.extend(embeddings_tensor.numpy().tolist())
        
    return embeddings

def generate_dates(start_date, end_date):
    """
        Generate the list of year month.

        Args:
            start_date: start date in the form of "year-month" eg. "2019-01".
            end_date: end date in the form of "year-month" eg. "2022-11".
        
        Return:
            dates: the list of year month between the start_date and end_date in datetime format.
    """
    # convert the strings into workable datetime datatype
    strt_dt = datetime.date(int(start_date[:4]),int(start_date[5:]),1)
    end_dt = datetime.date(int(end_date[:4]),int(end_date[5:]),1)

    # generate the list of year-month between the start and end date
    dates = [dt for dt in rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]
    return dates

def clean_dates(dates):
    """
        Clean the datetime entries in the list into the form of "year-month" eg. "2019-01".

        Args:
            dates: list of dates that we want to convert.
        
        Return:
            cleaned_dates: list of cleaned dates in the correct format.
    
    """
    cleaned_dates = list()
    for date in dates:
        if date.month<10: # if the month is single digit, add a '0'
            cleaned_dates.append(f"{date.year}-0{date.month}")
        else: # month is double digit
            cleaned_dates.append(f"{date.year}-{date.month}")
    
    return cleaned_dates

input_json = sys.argv[1]
with open(input_json, "r") as f:
    config = json.load(f)

inputs = config["generate_processed_data"]
start_date = inputs["start_date"]
end_date = inputs["end_date"]

# get the year-month in between the start and end date
dates = clean_dates(generate_dates(start_date, end_date))

# defining model file path
model_picklepath = inputs["model_picklepath"]
tokenizer_filepath = inputs["tokenizer_filepath"]

# extracting the model and tokenizer
with open(model_picklepath, "rb") as f:
    model = pickle.load(f)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_filepath)

mcf_ssoc_converter = pd.read_excel("SSOC Listing with classification_MOMCoLab.xlsx")

# every date input
for date in dates:
    folder_path = f"data/raw/{date}"

    # check if folder path is valid
    if not os.path.isdir(folder_path): # folder path is invalid
        logging.warning(f"Could not find data in {folder_path}, skipping to the next date.") 
        continue # go on to the next argument

    logging.info(f"Converting raw data for {date}")
    df = pd.DataFrame()
    index = 0 # track the indexing for pd.concat

    for file in os.listdir(folder_path):
        # if index == 128: # restrict the number of json files we want to read for testing, remove this in future
        #     break

        file_path = os.path.join(folder_path, file) # path to the json file
        f = open(file_path, )
        output, output_date = extract_mcf_data(json.load(f)) # extract the mcf data and convert to dictionary

        df = pd.concat([df, pd.DataFrame(output, index=[index])])
        index += 1

    # convert from MCF SSOC to SSOC2020
    df = mcfssoc_to_ssoc2020(mcf_ssoc_converter, df)

    # generate cleaned description and embeddings
    df = generate_cleaned_description(df)
    embeddings = generate_embeddings(model, tokenizer, df)

    # create a new column `yearMonth` to track the yearMonth of the data
    df['yearMonth'] = date 

    # creating the folder if does not exist
    if not os.path.isdir(f"data/processed/{date}"):
        logging.info(f"data/processed/{date} is not found in `processed`")
        logging.info(f"Creating data/processed/{date} in `proccessed`")
        os.mkdir(f"data/processed/{date}")

    # creating the embeddings pickle file
    with open(f"data/processed/{date}/embeddings.pickle", "wb") as f:
        pickle.dump(np.array(embeddings), f)

    # creating the csv file
    df.to_csv(f"data/processed/{date}/{date}.csv", index=False)