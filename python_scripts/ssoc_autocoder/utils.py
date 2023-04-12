import pandas as pd
import numpy as np
import copy

def verboseprint(verbose=True):
    """
    Allow verbosity for printing out functions

    Parameter:
        verbose (Boolean): Defaulted to True

    Returns:
        function with to print or not to print
    """
    return print if verbose else lambda *a, **k: None


def processing_raw_data(filename, *colnames):
    """
    Processes the raw dataset into the right data structure

    Parameters:
        filename (str): Link to the specific dataset
        *colnames (str): Subsequent columns in order. Job Title, Job Description and SSOC 2015

    Returns:
        processed_data(link, *colnames): The dataset is imported and processed

    Raises:
        AssertionError: If any of the colnames specified do not exist in the data
        AssertionError: If there is a null value in the data
    """

    # Reading in the CSV file
    print(f'Reading in the CSV file "{filename}"...')
    data = pd.read_csv(filename)

    # Checking that the colnames are entered in correctly
    print('Subsetting the data and renaming columns...')
    for colname in list(colnames):
        if colname not in data.columns:
            raise AssertionError(f'Error: Column "{colname}" not found in the CSV file.')

    # Subsetting the data to retain only the required columns
    data = data[list(colnames)]

    # Renaming the columns
    dict_map = {colnames[0]: 'Job_ID',
                colnames[1]: 'Title',
                colnames[2]: 'Description',
                colnames[3]: 'SSOC_2015'}
    data.rename(columns=dict_map, inplace=True)

    # To Ben: We shouldn't coerce to numeric as there are some SSOCs with characters
    #data['SSOC'] = pd.to_numeric(data['SSOC'], errors='coerce')
    #data['SSOC'] = data['SSOC'].astype(int)

    # Enforcing string type character for the SSOC field and doing a whitespace strip
    data['SSOC_2015'] = data['SSOC_2015'].astype('str').str.strip()

    # To Ben: This is unexpected behaviour - we should raise a warning/error instead of there are nulls.
    #data = data.dropna(axis=0, how='any')

    # Checking if there are any unexpected nulls in the data
    if np.sum(data.isna().values) != 0:
        raise AssertionError(f"Error: {np.sum(data.isna().values)} nulls detected in the data.")
    else:
        print('No nulls detected in the data.')

    print('Processing complete!')
    return data

def wpd_job_title_converter(SSOC_code, job_title_map):
    """
    Returns a mapped version of the predictions formatted, based on the job title mapping by WPD

    Parameters:
        predictions_formatted [predictions]: List of preditions by our model
        job_title_map (dic): SSOC to job titles mapping, one to many mapping

    Returns:
        job_occupation_listing [occupations]: A list of job titles based on WPD_Job_title_list_mapping and the corresponding SSOC
    """  

    return job_title_map[SSOC_code]["jobtitle"]
    
# Print iterations progress
# From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()