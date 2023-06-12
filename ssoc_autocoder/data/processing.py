import os
import json
import pandas as pd
from ssoc_autocoder.utils import printProgressBar

def extract_mcf_data(json):
    """
    Extracts the relevant data from the MyCareersFuture job ad JSON file

    Parameters:
        json (dict): JSON (loaded as a Python dictionary) to extract data from

    Returns:
        Standardised dictionary containing only the relevant data
    """
    
    # Initialising the output dictionary
    output = {}

    # Extracting general information of the job posting
    transfer = ['uuid', 'title', 'description', 'minimumYearsExperience', 'numberOfVacancies']
    for key in transfer:
        try:
            output[key] = json[key]
        except:
            # If keys not found, treat file as failure to extract
            return None, None

    # Extract skills, skills are mainly captured in separate JSON objects 
    output['skills'] = ', '.join([entry['skill'] for entry in json['skills']])
    
    # Extract hiring company
    company = ['name', 'description', 'ssocCode', 'ssicCode', 'employeeCount']
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
    metadata = ['originalPostingDate', 'newPostingDate', 'expiryDate', 'totalNumberOfView', 'totalNumberJobApplication']
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
    
    """

    # Initialising the output dictionary holding all the data
    output = {}

    # Iterating through all the JSON files in the folder
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for filename in os.listdir(path):    
        
        f = open(path + filename)
        entry = json.load(f)
        
        extracted_result, date = extract_mcf_data(entry)

        if extracted_result:
            date_year_mth = date[0:7]
            if date_year_mth in output: 
                output[date_year_mth].append(extracted_result)
            else:
                output[date_year_mth] = [extracted_result]
        else:
            print(f'{filename} has missing key values')
            fi = open("json_to_remove.txt", "a")
            fi.write(f'{filename}\n')
            fi.close()
    
    return output