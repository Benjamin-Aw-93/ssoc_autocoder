import os
import json
import pandas as pd
from .utils import verboseprint

# Load verbosity ideally should load in command line, write as -v tag in cmd
# Should load load at the start of the script
verbosity = False  # default value

verboseprinter = verboseprint(verbosity)

path = "../Data/Raw/"


def extract_mcf_data(json):
    """
    Extracting information in the json file and compartmentalise it into a dictionary, to be used in extract_and_split

    Parameters:
        json (dic): Extracted from MCF site

    Returns:
        A dictionary of extracted infomation, and the date of the post
    """

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
    metadata = ['originalPostingDate', 'newPostingDate', 'expiryDate',
                'totalNumberOfView', 'totalNumberJobApplication']
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


def extract_and_split(path):
    """
    Run extract_mcf_data on each txt file

    Parameters:
        path (str): Path of mcf_raw

    Returns:
        Return a dic, with key being the dates and the values being a list of extracted data
    """

    output = {}

    for filename in os.listdir(path + "mcf_raw"):

        verboseprinter(f'Reading in {filename}')
        f = open(path + "/mcf_raw/" + filename)
        entry = json.load(f)

        extracted_result, date = extract_mcf_data(entry)

        if extracted_result:
            date_year_mth = date[0:7]
            if date_year_mth in output:
                output[date_year_mth].append(extracted_result)
            else:
                output[date_year_mth] = [extracted_result]
        else:
            verboseprinter(f'{filename} has missing key values')
            fi = open(path + "json_to_remove.txt", "a")
            fi.write(f'{filename}\n')
            fi.close()

    return output
