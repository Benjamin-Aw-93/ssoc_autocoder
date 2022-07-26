import pickle
import os
import json
import pandas as pd
from .utils import verboseprint
import datetime


# Load verbosity ideally should load in command line, write as -v tag in cmd
# Should load load at the start of the script
verbosity = False  # default value

verboseprinter = verboseprint(verbosity)


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
    now = datetime.datetime.now()

    for filename in os.listdir(path):

        verboseprinter(f'Reading in {filename}')
        f = open(path+filename)
        entry = json.load(f)

        extracted_result, date = extract_mcf_data(entry)

        if extracted_result:
            
            #tag the JOB ID
            extracted_result["MCF_Job_Ad_ID"] = filename[:-5]

            #Tag the date when the json was converted to the csv
            extracted_result['JSON to CSV date'] = now
            
            #get the year of the Job_ID
            year = date[0:4]

            #get the month of the Job_ID
            month = date[5:7]

            #get the day of the Job_ID
            day = date[-2:]

            #get the week of the Job_ID so we can group it in weeks
            week_num = datetime.date(int(year), int(month), int(day)).isocalendar()[1]
            
            #if it is a single digit week add a 0 infront for filename
            if week_num <10:
                date_year_week= year+'-'+'0'+str(week_num)
            else:
                date_year_week= year+'-'+str(week_num)
                

            if date_year_week in output: 
                output[date_year_week].append(extracted_result)
            else:
                output[date_year_week] = [extracted_result]
        else:
            verboseprinter(f'{filename} has missing key values')

    return output

