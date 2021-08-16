import pandas as pd
import re
import spacy
# Functions use

# Reading in files and selecting + renaming columns


def reading_selecting_data(link, *colnames):
    '''

    Returns the dataset with specific colnames.

    Parameters:
        link (str): Link to the specific dataset
        *colnames: Subsequent columns in order. Job Title, Job Description and SSIC

    Returns:
        subseted_data(link, *colnames): The dataset is read in and selected. All rows with Nan values are dropped.

    '''
    data = pd.read_csv(link)
    data = data[list(colnames)]
    dict_map = {colnames[0]: 'Title',
                colnames[1]: 'Description',
                colnames[2]: 'SSOC'}
    data.rename(columns=dict_map, inplace=True)

    data['SSOC'] = pd.to_numeric(data['SSOC'], errors='coerce')

    data = data.dropna(axis=0, how='any')

    data['SSOC'] = data['SSOC'].astype(int)

    return data


def remove_html_tags_newline(text):
    '''

    Remove html tags from a string with generic regex

    Parameters:
        text (str): Selected text

    Returns:
        cleaned_text(text) : Text with html tags and new line  removed

    '''

    clean = re.compile('<.*?>')
    newline_clean = re.compile('\n')
    return re.sub(newline_clean, ' ', re.sub(clean, '', text)).lower()


mcf_df = reading_selecting_data(
    "..\Data\Raw\WGS_Dataset_Part_1_JobInfo.csv", "title", "description", "ssoc_code")

# Apply removal across rows along both the Title and Description
mcf_df['Title'] = mcf_df['Title'].apply(remove_html_tags_newline)

mcf_df['Description no HTML'] = mcf_df['Description'].apply(remove_html_tags_newline)

# Loding spacy, pipline for further cleaning
nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])


def to_doc(text):
    '''
    Create SpaCy documents by wraping text with pipline function
    '''
    return nlp(text)


def lemmatize_remove_stop(doc):
    '''
    Take the `token.lemma_` of each non-stop word
    '''
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]


# create documents for all tuples of tokens
docs = list(map(to_doc, mcf_df['Description no HTML']))

# apply stop word removal and lemmatization to each text within Description
mcf_df['Lem Desc rm stop words tokenised'] = list(map(lemmatize_remove_stop, docs))

# Exploring extraction of job description using HTML tags
mcf_df.to_csv("..\Data\Processed\WGS_Dataset_JobInfo_precleaned.csv", index=False)
