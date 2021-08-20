# Importing required libraries
import numpy as np
import pandas as pd
import re

# Try to get summarised data out using HTML tags

mcf_df = pd.read_csv("..\Data\Processed\WGS_Dataset_JobInfo_precleaned.csv")

# Temp: Make dataset smaller so that it is easier to work with
mcf_df.head()

mcf_df = mcf_df[["Title", "Description", "SSOC"]].sample(frac=0.1)


# Naive way of extracting
def extracting_job_desc_naive(text):
    '''

    Extract job description using <li> HTML tags

    Parameters:
        text (str): Selected text

    Returns:
        list_extracted_text(text) : Text with html tags <li>

    '''

    pattern = re.compile('(?<=<li>).*?(?=</li)')
    return pattern.findall(text)


mcf_df['Description naive'] = mcf_df['Description'].apply(extracting_job_desc_naive)

mcf_df['Count'] =
