# Importing libraries
import pandas as pd
import re
import numpy as np
# import spacy

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

def main():

    # Importing and processing the raw data
    mcf_df = processing_raw_data("..\Data\Raw\WGS_Dataset_Part_1_JobInfo.csv", "job_post_id", "title", "description", "ssoc_code")

    # Apply removal across rows along both the Title and Description
    mcf_df['Title'] = mcf_df['Title'].apply(remove_html_tags_newline)
    mcf_df['Description no HTML'] = mcf_df['Description'].apply(remove_html_tags_newline)

    # To Ben: Not used, we can drop it for now
    # Loading spacy, pipeline for further cleaning
    # nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])
    # create documents for all tuples of tokens
    # docs = list(map(to_doc, mcf_df['Description no HTML']))
    # apply stop word removal and lemmatization to each text within Description
    #mcf_df['Lem Desc rm stop words tokenised'] = list(map(lemmatize_remove_stop, docs))

    # Exploring extraction of job description using HTML tags
    mcf_df.to_csv("..\Data\Processed\WGS_Dataset_JobInfo_precleaned.csv", index=False)

if __name__ == "__main__":
    main()

## Currently inactive, keeping for reference

def remove_html_tags_newline(text):
    """
    Removes HTML and newline tags from a string with generic regex

    Parameters:
        text (str): Selected text

    Returns:
        cleaned_text(text) : Text with html tags and new line removed
    """

    clean = re.compile('<.*?>')
    newline_clean = re.compile('\n')
    return re.sub(newline_clean, ' ', re.sub(clean, '', text)).lower()

def to_doc(text):
    """
    Create SpaCy documents by wrapping text with pipeline function
    """
    return nlp(text)

def lemmatize_remove_stop(doc):
    '''
    Take the `token.lemma_` of each non-stop word
    '''
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
