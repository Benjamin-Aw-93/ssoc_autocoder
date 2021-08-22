# Importing required libraries
import numpy as np
import pandas as pd
import re

# Try to get summarised data out using HTML tags

mcf_df = pd.read_csv("..\Data\Processed\WGS_Dataset_JobInfo_precleaned.csv")

# Temp: Make dataset smaller so that it is easier to work with
mcf_df.head()

mcf_df = mcf_df[["Title", "Description", "SSOC"]].sample(frac=0.1)


# Create a class that takes in the dataset and the cleaning function
# Captures all the necessary information:
# 1. Number of entries filled
# 2. Take the top n as a sample to be returned

class extraction_text:
    def __init__(self, df_text, cleaning_fn):
        self.text = df_text
        self.extracted_text = self.text.apply(extracting_job_desc_naive)
        self.percentage_completed = sum(1 if isinstance(
            text, list) else 0 for text in self.extracted_text) / self.extracted_text.size
        self.subsample = df_text.head(100)


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


def main():
    naive_extraction_obj = extraction_text(mcf_df["Description"], extracting_job_desc_naive)


if __name__ == "__main__":
    main()
