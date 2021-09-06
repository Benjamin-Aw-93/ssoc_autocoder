# Importing required libraries
import numpy as np
import pandas as pd
import re
import spacy
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
                 "Outlines", "Outline", "Role", "Roles",
                 "Responsibility", "Responsibilities"]

    output = []

    for word in lst_words:
        pattern = re.compile(r"(?smix)(?<=" + word + ").*?(?=<strong>)")
        search = pattern.findall(text)
        output.append(search)

    flat_list = [remove_html_tags_newline(item) for sublist in output for item in sublist]

    clean = re.compile(r'(?m)\s+.*\w')

    return [' '.join(clean.search(text).group(0).strip().split()) for text in flat_list]

# Extracting based on ul tag to get title and description


def max_similarity(lst, word):
    return max(map(lambda x: (nlp(x).similarity(nlp(word))), lst))


def lemmatize_remove_punct(doc):
    '''
    Take the `token.lemma_` of each non-stop word
    '''
    return [token.lemma_ for token in doc if not token.is_punct]


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
    splitlst = [t.split(r'</p>') for t in textlst]

    splitlst = []

    for t in textlst:
        txt = t.split(r'</p>')
        if len(txt) == 2:
            splitlst.append(txt)
        else:
            placeholder = []
            for iter in txt[::-1]:
                if "strong" not in iter:
                    placeholder.append(iter)
                else:
                    placeholder.append(iter)
                    splitlst.append(placeholder[::-1])
                    placeholder = []

    splitdic = [{'title': lst[0], 'description': ' '.join(lst[1:])} for lst in splitlst]

    # cleaning each individual items
    splitdic_cleaned = [{'title': " ".join(remove_html_tags_newline(dic['title']).split()),
                         'description': " ".join(remove_html_tags_newline(dic['description']).split())} for dic in splitdic]

    if len(splitdic_cleaned) == 0:
        return []

    deci_table = [lemmatize_remove_punct(nlp(dic['title'])) for dic in splitdic_cleaned]

    try:
        deci_table_index = [max_similarity(lst, "description") for lst in deci_table]
        deci_table_index = deci_table_index.index(max(deci_table_index))

    except ValueError:
        return []

    else:
        return splitdic_cleaned[deci_table_index]


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


# testing out extraction
text = '<p><strong>Role Descriptions:</strong></p> <p>This position reports to the Commercial Planning &amp; Operations (CPO) manager, and interfaces with OBFS (Own Brands &amp; Food Solutions) staff and product managers.</p> <p><br></p> <p><strong>Specific Responsibilities:</strong></p> <p>Individual will be involved in:</p> <ol> <li>Sourcing process - building long list of suppliers, preparing RFI, RFP, contacting suppliers for the roll out of existing and new products</li> <li>Conducting market-price research, generating price-match proposals for Own Brand portfolio</li> </ol> <p><strong>Technical Skills and Competencies:</strong></p> <ul> <li>Project management, including how to engage stakeholders professionally to deliver project targets</li> <li>Active participation in the sourcing process</li> <li>Participation in price planning process, including contributing to and pitching price-match proposals to stakeholders</li> </ul> <p><strong>Duration of Traineeships:</strong></p> <ul> <li>6 Months</li> </ul> <p><strong>Approved Training Allowance:</strong></p> <ul> <li>Fresh Graduates: S$2500</li> <li>Non-Mature Mid-Career Individuals: $2800</li> <li>Mature Mid-Career Individuals: $3200</li> </ul> <p>This position is open for both recent graduates and mid-career individuals (mature and non mature). Graduates interested in this position should possess a University Degree Qualification. Mid-career individuals from any qualification level can apply.</p></div>'


extracting_job_desc_ultag(text)


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

    output_df_text = pd.DataFrame.from_dict(output_dic_txt)

    output_df_text.to_csv(
        "..\Data\Processed\Artifacts\Extracted_Text_Sample_100.csv", index=False)

    print("Done with testing")


if __name__ == "__main__":
    main()
