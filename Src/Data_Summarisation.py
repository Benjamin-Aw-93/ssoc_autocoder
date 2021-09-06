# Importing required libraries
import numpy as np
import pandas as pd
import re
import spacy
import string
from Data_Cleaning import remove_html_tags_newline, to_doc, lemmatize_remove_stop
from spacy.matcher import Matcher
from collections import deque

# Try to get summarised data out using HTML tags

mcf_df = pd.read_csv("..\Data\Processed\WGS_Dataset_JobInfo_precleaned.csv")

# Temp: Make dataset smaller so that it is easier to work with
mcf_df.head()

mcf_df = mcf_df[["Job_ID", "Title", "Description", "SSOC"]].sample(frac=0.1)

# Create a class that takes in the dataset and the cleaning function
# Captures all the necessary information:
# 1. Number of entries filled
# 2. Take a random sample of 100 to be returned


class extraction_text:
    def __init__(self, df_text, cleaning_fn):
        self.text = df_text
        self.extracted_text = self.text.apply(cleaning_fn)
        self.percentage_completed = sum(1 if bool(
            text) else 0 for text in self.extracted_text) / self.extracted_text.size
        self.subsample_uncleaned = df_text.sample(n=100, random_state=20)  # subsample set seed
        self.subsample_cleaned = self.extracted_text.sample(n=100, random_state=20)


# Creating nlp pipeline
nlp = spacy.load('en_core_web_lg')

# Naive way of extracting


def extracting_job_desc_naive(text):
    '''

    Extract job description using <li> HTML tags

    Parameters:
        text (str): Selected text

    Returns:
        list_extracted_text(text) : Text with html tags <li>

    '''

    pattern = re.compile(r'(?smix)(?<=<li>).*?(?=</li>)')
    return pattern.findall(text)

# Extracting based on finding the word description


def extracting_job_desc_named(text):
    '''

    Extract job description using specific header text

    Parameters:
        text (str): Selected text

    Returns:
        list_extracted_text(list[text]): Extracted text

    '''
    # List of header words to look out for
    lst_words = ["Descriptions", "Description", "Competencies", "Competency",
                 "Responsibility", "Responsibilities", "Duty", "Duties",
                 "Outlines", "Outline", "Role", "Roles"]

    output = []

    for word in lst_words:
        pattern = re.compile(r"(?smix)(?<=" + word + ").*?(?=<strong>)")
        search = pattern.findall(text)
        output.append(search)

    flat_list = [remove_html_tags_newline(item) for sublist in output for item in sublist]

    return [' '.join(text.strip(string.punctuation).strip().split()) for text in flat_list]


def max_similarity(lst, word):
    return nlp(' '.join(lst)).similarity(nlp(word))


def lemmatize_remove_punct(doc):
    '''
    Take the `token.lemma_` of each non-stop word
    '''
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]


# Extracting based on ul tag to get title and description
def extracting_job_desc_ultag(text):
    '''

    Extract job description using specific header text

    Parameters:
        text (str): Selected text

    Returns:
        list_extracted_text(list[text]): Extracted text

    '''
    pattern = re.compile(r'(?smix)(?=<p>).*?(?<=</p>).*?(?=<ol>|<ul>).*?(?<=</ol>|</ul>)')
    textlst = pattern.findall(text)

    splitlst = []

    for t in textlst:
        txt = t.split(r'</p>')
        if len(txt) == 2:  # only 2 items, so first must be the header, the second must be the description
            splitlst.append(txt)
        else:
            placeholder = []  # palceholder list, to be added to the main list
            for iter in txt[::-1]:  # go through the back of the list
                if "<p>" not in iter:  # if paragraph tag not in iter
                    placeholder.append(iter)  # append as description
                else:
                    placeholder.append(iter)  # append as title
                    # reverse the list so that the title is now at position 0
                    splitlst.append(placeholder[::-1])
                    placeholder = []  # reset the list

    splitdic = [{'title': lst[0], 'description': ' '.join(lst[1:])} for lst in splitlst]

    # cleaning each individual items
    splitdic_cleaned = [{'title': " ".join(remove_html_tags_newline(dic['title']).split()),
                         'description': " ".join(remove_html_tags_newline(dic['description']).split())} for dic in splitdic]

    # return nothing
    if len(splitdic_cleaned) == 0:
        return []

    deci_table = [lemmatize_remove_punct(nlp(dic['title'])) for dic in splitdic_cleaned]
    # if fail due to non-existing title, return empty
    try:
        deci_table_index = [max_similarity(
            lst, "job description, duty and responsibility") for lst in deci_table]

        deci_table_withscore = list(zip(splitdic_cleaned, deci_table_index))

        # entry[0] being the dictionary, entry[1] being the score, rewriting the zip
        output = []

        for entry in deci_table_withscore:
            temp_dic = entry[0]
            temp_score = entry[1]
            temp_dic['score'] = temp_score
            output.append(temp_dic)

    except ValueError:
        return []

    else:
        return output

# Trying to match using POS tags (KIV)

# def extracting_job_desc_POS(text):
#     '''
#
#     Extracting job description using POS (verb)
#
#     Parameters:
#         text (str): Selected text
#
#     Returns:
#         list_extracted_text(text): Extracted text
#
#     '''
#
#     doc = nlp(text)
#     print([token.text for token in doc])
#     matcher = Matcher(nlp.vocab)
#     pattern = [{'POS': 'VERB', 'OP': '+'}, {'TEXT': '<', 'OP': '*'}]
#
#     matcher.add('ALL_ACTIVITY_PATTERN', [pattern])
#
#     matches = matcher(doc)
#     matches
#     doc[11:12]
#     doc[58:69]
#     doc[70:75]


def main():

    naive_extraction_obj = extraction_text(mcf_df["Description"], extracting_job_desc_naive)
    print(
        f'Extracted naive_extraction_obj, percentage extracted: {naive_extraction_obj.percentage_completed}')

    named_extraction_obj = extraction_text(mcf_df["Description"], extracting_job_desc_named)
    print(
        f'Extracted named_extraction_obj, percentage extracted: {named_extraction_obj.percentage_completed}')

    ultag_extraction_obj = extraction_text(mcf_df["Description"], extracting_job_desc_ultag)
    print(
        f'Extracted ultag_extraction_obj, percentage extracted: {ultag_extraction_obj.percentage_completed}')

    lst_of_extraction_obj = dir()

    # write out to dic, change dic to table

    output_dic_per = {}

    for var_str in lst_of_extraction_obj:
        output_dic_per[var_str] = eval(var_str).percentage_completed

    output_df_stats = pd.DataFrame(output_dic_per.items(), columns=[
        'Extraction_Method', 'Percentage_Captured'])

    output_df_stats.to_csv("..\Data\Processed\Artifacts\Extracted_Percentages.csv", index=False)

    output_dic_txt = {'Raw_text': eval(lst_of_extraction_obj[0]).subsample_uncleaned}

    for var_str in lst_of_extraction_obj:
        output_dic_txt[var_str] = eval(var_str).subsample_cleaned

    output_dic_txt['ultag_extraction_max'] = max(
        ultag_extraction_obj.subsample_cleaned, key=lambda x: x['score'])

    output_df_text = pd.DataFrame.from_dict(output_dic_txt)

    output_df_text.to_csv(
        "..\Data\Processed\Artifacts\Extracted_Text_Sample_100.csv", index=False)

    print("Done with testing")


if __name__ == "__main__":
    main()
