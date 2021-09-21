import pandas as pd
import re
import numpy as np
from pathlib import Path


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


def remove_prefix(text, prefixes):
    """
    Checks if the first text begins with a certain prefix

    Parameters:
        text (str): Text to check for
        prefixes [str]: List of prefixes to check from

    Returns:
        Truncated text, or text if no prefix available
    """
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def check_if_first_word_is_verb(string):
    """
    Checks if the first word of the string is a verb

    Parameters:
        string (str): Text to check for

    Returns:
        Whether the first word is a verb or not
    """

    # Define some words that should be False
    # regardless of what Spacy says
    override_false_list = ['proven', 'possess']

    # Define some words that should be True
    # regardless of what Spacy says
    override_true_list = ['review', 'responsible', 'design', 'to', 'able']

    # If the string is a zero length, return False
    if len(string) == 0:
        return False

    # If the first word is in the override false list, return False
    if string.split(' ')[0].lower() in override_false_list:
        return False

    # If the first word is in the override True list, return True
    if string.split(' ')[0].lower() in override_true_list:
        return True

    # If the first two words are "you are", we truncate it
    string = remove_prefix(string.lower(), ['you are'])

    # Check if the first word is a verb
    return nlp(string)[0].pos_ == 'VERB'


def clean_raw_string(string):
    """
    Cleans the raw text from problematic strings or abbreviations

    Parameters:
        string (str): Text to clean for

    Returns:
        Cleaned text without problematic strings or abbreviations
    """

    # Identify some common problematic strings to remove
    to_remove = ['\n', '\xa0', '&nbsp;', '&amp;', '\t', '&rsquo;']

    # Remove these strings
    for item in to_remove:
        string = string.replace(item, '')

    # Identify some common abbreviations to replace
    to_replace = [('No.', 'Number')]

    # Replace these strings
    for item1, item2 in to_replace:
        string = string.replace(item1, item2)

    # Remove all non-unicode characters
    # Deprecated due to reliance on bullet points
    # string = ''.join([i if ord(i) < 128 else ' ' for i in string])

    return string


def clean_html_unicode(string):
    """
    Cleans the raw text from html codes

    Parameters:
        string (str): Text to clean for

    Returns:
        Cleaned text without problematic strings or abbreviations
    """
    # Initialise the output string
    cleaned_string = string

    # This is run in order, so be careful!
    # <.*?>: removes html tages
    # ^\d+\.{0,1} removes any bullet points for digits
    # [^\w\s] removes any other symbols
    cleaning_regex = ['<.*?>', '^\d+\.{0,1}', '[^\w\s]']

    # Iteratively apply each regex
    for regex in cleaning_regex:
        cleaned_string = re.sub(regex, '', cleaned_string).strip()

    return cleaned_string


def check_list_for_verbs(list_elements):
    """
    Check list for verbs after extracting text text based on each method below

    Parameters:
        list_elements [str]: list of text produced by each function

    Returns:
        list of text of maximum verb score, else empty list
    """
    # Initialise a list to store the output
    verb_scores = []

    # Iterate through each of the list elements passed in
    for list_element in list_elements:

        # Use regex to split up the list into items
        # Note this depends on whether the list elements
        # passed in are lists (ol/ul) or paragraph lists (p)
        if list_element[0:4] in ['<ul>', '<ol>']:
            list_items_pattern = re.compile(r'(?=<li>).*?(?<=</li>)')
        else:
            list_items_pattern = re.compile(r'(?=<p>).*?(?<=</p>)')

        # Split each list up into the constituent items
        list_items = list_items_pattern.findall(list_element)

        # Initialise a count of number of items beginning with a verb
        count = 0

        # Iterate through each item in the list
        for list_item in list_items:

            # Remove all the HTML tags and check if the first word is a verb
            list_item = clean_html_unicode(list_item)
            #list_item = re.sub("[^\w\s]", "", re.sub('^\d+\.{0,1}', '', re.sub('<.*?>', '', list_item.replace('\t', '')).strip())).strip()

            # Check if the first word is a verb, and add to score if it is
            if check_if_first_word_is_verb(list_item):
                count += 1

        # Add the list length and verb score to the output
        verb_scores.append((len(list_items), count/len(list_items)))

    # Initialise the list to store the new set of
    # list elements which we are merging if they
    # are short lists with lots of verbs
    for_recursive = []

    # Iterating over each verb score
    for i, verb_score in enumerate(verb_scores):

        # Always append the first list item
        if i == 0:
            for_recursive.append(list_elements[i])

        # For other items, check if there are less than 6 items
        # and the verb score is at least 70%. If so, we merge it
        elif (verb_score[0] < 6) and (verb_score[1] >= .7):

            # Remove the starting list tag if it is included
            list_elements_i_cleaned = re.sub(r'(<ul>|<ol>|</ul>|</ol>)', '', list_elements[i])

            # If the preceding list has a </ul> or </ol> tag
            # then we should remove it before concatenating the
            # strings, but otherwise we just concat the strings directly
            if for_recursive[-1][:-5] in ['</ul>', '</ol>']:
                for_recursive[-1] = for_recursive[-1][:-5] + " " + \
                    list_elements_i_cleaned + for_recursive[-1][-5:]
            else:
                for_recursive[-1] += list_elements[i]

        # Otherwise we just append it back to the list
        else:
            for_recursive.append(list_elements[i])

    # Run the recursive function if we have merged some lists together
    if len(for_recursive) != len(list_elements):
        return check_list_for_verbs(for_recursive)

    # Otherwise, we output the verb scores
    else:

        # Append the verb score to the list
        # with a exception for very short lists
        final_verb_scores = []
        for verb_score in verb_scores:

            # If the length is less than 3
            if verb_score[0] < 3:
                final_verb_scores.append(min(verb_score[1], 0.5))  # cap the score at 50%
            else:
                final_verb_scores.append(verb_score[1])

        # Return the list with maximum verb score, assuming at least
        # 50% of the list contains verbs
        if max(final_verb_scores) >= 0.5:
            return list_elements[final_verb_scores.index(max(final_verb_scores))]
        else:
            return []


def process_li_tag(text):
    """
    Process job descriptions using li tags

    Parameters:
        text: Job descriptions text

    Returns:
        List of extracted text, post-processed by check_list_for_verbs(list_elements)
    """

    # Extract all lists in the HTML with a list tag (<ol> or <ul>)
    # Regex explanation:
    # (?=<ol>|<ul>) is the lookahead for the <ol> or <ul> tag
    # .* captures everything between the tags, ? restricts it to capturing one set only
    # (?<=</ol>|</ul>) is the lookbehind for the </ol> or </ul> tag
    list_pattern = re.compile(r'(?=<ol>|<ul>).*?(?<=</ol>|</ul>)')
    list_elements = list_pattern.findall(text)

    if len(list_elements) == 0:
        return []

    return check_list_for_verbs(list_elements)


def process_p_list(text):
    """
    Process job descriptions using p tags. Extracting out text preceeded by literal bulletpoints or numeric points.

    Parameters:
        text: Job descriptions text

    Returns:
        List of extracted text, post-processed by check_list_for_verbs(list_elements)
    """
    # Extract all lists in the HTML with a paragraph tag (<p>)
    # Regex explanation:
    # (?=<p>) is the lookahead for the <p> tag
    # .* captures everything between the tags, ? restricts it to capturing one set only
    # (?<=</p>) is the lookbehind for the </p> tag
    para_pattern = re.compile(r'(?=<p>).*?(?<=</p>)')
    para_elements = para_pattern.findall(text)

    # Check for specific unicode characters that can be used as bullet points
    unicode_to_check = ['\u2022', '\u002d', '\u00b7']
    bullet_pt_presence = []
    for para_element in para_elements:

        # Remove all the HTML tags
        para_element_cleaned = re.sub('<.*?>', '', para_element).strip()

        # Check if the string is non-empty
        if len(para_element_cleaned) > 0:

            # Check if the first character has any bullet points
            result1 = para_element_cleaned[0] in unicode_to_check

            # Check if the first character is a numbered list
            # by checking if the re.match() returns anything
            result2 = re.match(r'^\d+\.', para_element_cleaned) is not None

            bullet_pt_presence.append(result1 or result2)

        # If it is empty, then it doesn't contain any bullet points
        else:
            bullet_pt_presence.append(False)

    # Initialise the lists
    output = []
    p_list = []

    # Build an equivalent list of list items by iterating
    # through the boolean list indicating if there is
    # a bullet point character at the start of the string
    for i, value in enumerate(bullet_pt_presence):

        # If there is a bullet point character
        if value:

            # Append the string to the para list
            p_list.append(para_elements[i])

        # If there is no bullet point character
        else:

            # Append the para list if it is non-empty
            if len(p_list) > 0:
                output.append(' '.join(p_list))

            # Reset the para list
            p_list = []

    if len(output) == 0:
        return []

    return check_list_for_verbs(output)


def process_p_tag(text):
    """
    Process job descriptions using p tags. Extracting out text preceeded a verb

    Parameters:
        text: Job descriptions text

    Returns:
        List of extracted text, post-processed by check_list_for_verbs(list_elements)
    """
    # Extract all lists in the HTML with a paragraph tag (<p>)
    # Regex explanation:
    # (?=<p>) is the lookahead for the <p> tag
    # .* captures everything between the tags, ? restricts it to capturing one set only
    # (?<=</p>) is the lookbehind for the </p> tag
    para_pattern = re.compile(r'(?=<p>).*?(?<=</p>)')
    para_elements = para_pattern.findall(text)

    if len(para_elements) == 0:
        return []

    # Iterate through each paragraph element to see which one starts
    # with a verb, and we keep that paragraph element
    output = []
    for para_element in para_elements:

        # Remove all the HTML tags and check if the first word is a verb
        para_element_cleaned = re.sub("[^\w\s]", "", re.sub('<.*?>', '', para_element)).strip()
        if len(para_element_cleaned) > 0:
            if check_if_first_word_is_verb(para_element_cleaned):
                output.append(para_element)

    return " ".join(output)


def process_text(raw_text):
    """
    Process job description, put text through each process function, and return results according to precedence.

    Parameters:
        text: Job descriptions text

    Returns:
        Extracted text
    """
    # Remove problematic characters
    text = clean_raw_string(raw_text)

    li_results = process_li_tag(text)
    p_list_results = process_p_list(raw_text)
    p_results = process_p_tag(text)

    if len(li_results) > 0:
        print('List object detected')
        return li_results
    elif len(p_list_results) > 0:
        print('Paragraph list detected')
        return p_list_results
    elif len(p_results) > 0:
        print('Paragraphs detected')
        return p_results
    else:
        print('None detected, returning all')
        return re.sub('<.*?>', ' ', text)
