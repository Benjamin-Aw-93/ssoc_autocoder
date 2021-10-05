# import packages
import json
import re
import copy
from bs4 import BeautifulSoup
from utils import verboseprint
import sys

# load spacy object: To remove after testing
import spacy
nlp = spacy.load('en_core_web_lg')

# Load verbosity ideally should load in command line, write as -v tag in cmd
# Should load load at the start of the script
if (len(sys.argv) == 0):
    verbosity = True  # Default verbosity
else:
    if '-v' in sys.argv:
        verbosity = True
    else:
        verbosity = False

verboseprinter = verboseprint(verbosity)


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


def check_if_first_word_is_verb(string, nlp):
    """
    Checks if the first word of the string is a verb

    Parameters:
        string (str): Text to check for
        nlp (obj): Spacy nlp object

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
    string = remove_prefix(string.lower(), ['you are', 'are you'])

    # If trucation turns string empty, then return false
    if len(string) == 0:
        return False

    # Check if the first word is a verb
    return nlp(string)[0].pos_ == 'VERB'


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
    cleaning_regex = [r'<.*?>', r'^\d+\.{0,1}', r'[^\w\s]']

    # Iteratively apply each regex
    for regex in cleaning_regex:
        cleaned_string = re.sub(regex, '', cleaned_string).strip()

    return cleaned_string


def check_list_for_verbs(list_elements, nlp):
    """
    Check list for verbs after extracting text text based on each method below

    Parameters:
        list_elements [str]: list of bs4 tags produced by each function

    Returns:
        list of text of maximum verb score, else empty list
    """
    # Initialise a list to store the output
    verb_scores = []

    # Iterate through each of the list elements passed in
    for list_element in list_elements:
        # Use bs4 to split up the list into items
        # Note this depends on whether the list elements
        # passed in are lists (ol/ul) or paragraph lists (p)
        if list_element.name in ('ol', 'ul'):
            tag_of_interest = 'li'
        else:
            tag_of_interest = 'p'

        # Split each list up into the constituent items
        list_items = list_element.find_all(tag_of_interest)

        # Convert the list of bs4.tags back to list of string
        list_items = [str(item) for item in list_items]

        # Initialise a count of number of items beginning with a verb
        count = 0

        # Iterate through each item in the list
        for list_item in list_items:

            # Remove all the HTML tags and check if the first word is a verb
            list_item = clean_html_unicode(list_item)
            # list_item = re.sub("[^\w\s]", "", re.sub('^\d+\.{0,1}', '', re.sub('<.*?>', '', list_item.replace('\t', '')).strip())).strip()

            # Check if the first word is a verb, and add to score if it is
            if check_if_first_word_is_verb(list_item, nlp):
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
            # list_elements_i_cleaned = list_elements[i].find_all('li')
            # print(list_elements_i_cleaned)

            # If the preceding list has a </ul> or </ol> tag
            # then we should remove it before concatenating the
            # strings, but otherwise we just concat the strings directly
            if for_recursive[-1].name in ['ul', 'ol']:
                # Get name of the head tag ul or ol
                tag_header = for_recursive[-1].name
                # Find all li tags in both tags
                temp_list = for_recursive[-1].find_all('li') + list_elements[i].find_all('li')
                # Concat as string, wrapping with orginal head tag
                temp_list = '<%s>%s</%s>' % (tag_header,
                                             ' '.join([str(tag) for tag in temp_list]), tag_header)

                # Insert back to list as bs tag
                for_recursive[-1] = BeautifulSoup(temp_list, "html.parser").find(['ol', 'ul'])

            else:
                # Find all content in each tag
                temp_str = str(for_recursive[-1]) + " " + str(list_elements[i])

                for_recursive[-1] = BeautifulSoup(temp_str, "html.parser")

        # Otherwise we just append it back to the list
        else:

            for_recursive.append(list_elements[i])

    # Run the recursive function if we have merged some lists together
    if len(for_recursive) != len(list_elements):
        return check_list_for_verbs(for_recursive, nlp)

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


def process_li_tag(text, nlp):
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
    list_elements = BeautifulSoup(text, 'html.parser').find_all(['ol', 'ul'])

    # Deprecated: regex function
    # list_pattern = re.compile(r'(?=<ol>|<ul>).*?(?<=</ol>|</ul>)')
    # list_elements = list_pattern.findall(text)

    if len(list_elements) == 0:
        return []

    return check_list_for_verbs(list_elements, nlp)


def process_p_list(text, nlp):
    """
    Process job descriptions using p tags. Extracting out text preceeded by literal bulletpoints or numeric points.

    Parameters:
        text: Job descriptions text

    Returns:
        List of extracted text, post-processed by check_list_for_verbs(list_elements)
    """
    # Extract all lists in the HTML with a paragraph tag (<p>)
    # Only tags containing a bullet point (captured by unicode) or tags with Numbers followed by a period will be captured
    soup = BeautifulSoup(text, 'html.parser')
    output = []
    temp = []

    # Extract consecutive list of para
    for para in soup.select('p'):
        if re.match('^ *(\u2022|\u002d|\u00b7|\d+\.*)', str(para.contents[0])):
            # If match a bullet point or numeric numbering
            temp.append(para)
        else:
            if temp:
                # Adding bs tag item
                temp = [str(i) for i in temp]
                output.append(BeautifulSoup(' '.join(temp), 'html.parser'))
            temp = []

    # Final check
    if temp:
        temp = [str(i) for i in temp]
        output.append(BeautifulSoup(' '.join(temp), 'html.parser'))

    if len(output) == 0:
        return []

    return check_list_for_verbs(output, nlp)


def process_p_tag(text, nlp):
    """
    Process job descriptions using p tags. Extracting out text preceeded a verb

    Parameters:
        text: Job descriptions text

    Returns:
        List of extracted text, post-processed by check_list_for_verbs(list_elements)
    """
    # Extract all lists in the HTML with a paragraph tag (<p>)
    para_elements = BeautifulSoup(text, 'html.parser').find_all('p')

    if len(para_elements) == 0:
        return []

    # Iterate through each paragraph element to see which one starts
    # with a verb, and we keep that paragraph element
    output = []
    for para_element in para_elements:

        # Remove all the HTML tags and check if the first word is a verb
        para_element_cleaned = re.sub(r"[^\w\s]", "", re.sub(
            r'<.*?>', '', str(para_element))).strip()

        if len(para_element_cleaned) > 0:
            if check_if_first_word_is_verb(para_element_cleaned, nlp):
                output.append(para_element)

    output = [str(out) for out in output]

    return ' '.join(output)


def clean_raw_string(string):
    """
    Cleans the raw text from problematic strings or abbreviations

    Parameters:
        string (str): Text to clean for

    Returns:
        Cleaned text without problematic strings or abbreviations
    """

    # Identify some common problematic strings to remove
    to_remove = ['\n', '\xa0', '\t', '']

    # Remove these strings
    for item in to_remove:
        string = string.replace(item, '')

    # Identify some common abbreviations to replace
    to_replace = [('No.', 'Number')]

    # Replace these strings
    for item1, item2 in to_replace:
        string = string.replace(item1, item2)

    return string


def final_cleaning(processed_text):
    """
    Process extracted text final cleaning

    Prarmeters:
        processed_text (str, BS Tag) : Extacted text from each extraction function

    Returns:
        Final string that is cleaned
    """
    # Since text is a BS tag, cast it back into a string
    processed_text = str(processed_text)

    # Replace any <br> tags as fullstops
    processed_text = re.sub('<br>', '.', processed_text)

    # Replace any &amp;
    processed_text = re.sub('&amp;', '&', processed_text)

    # Replace any &nbsp;
    processed_text = re.sub('&nbsp;', ' ', processed_text)

    # Replace any &rsquo; and &lsquo;
    processed_text = re.sub('&rsquo;|&lsquo;', '\'', processed_text)

    # Replace any &ldquo; and &rdquo;
    processed_text = re.sub('&ldquo;|&rdquo;', '\'', processed_text)

    # Replace ’ with '
    processed_text = re.sub('\u2019|\u2018', '\'', processed_text)

    # Remove any special characters at the beginning of each statement
    processed_text = re.sub('(?<=>)\s*(\u2022|\u002d|\u00b7|\d+\.)', '.', processed_text)

    # Using regex, we remove all possible tags
    # Tags with slashes are replaces with period
    # Tags without slashes are replaced with spaces
    processed_text = re.sub('</.*?>', '.', processed_text)
    processed_text = re.sub('<.*?>', ' ', processed_text)

    # Remove inproper use of ;
    processed_text = re.sub(';', '.', processed_text)

    # To get proper text, we split the text up and join it back again
    processed_text = ' '.join(processed_text.split())

    # Remove if paragraph starts with punctuation
    processed_text = re.sub('^\s*[?.,!]\s*', '', processed_text)

    # If there are spaces betweens periods, remove them
    processed_text = re.sub('(?<=\.)\s*(?=\.)', '', processed_text)

    # If there are spaces between character and punctuation, remove them
    processed_text = re.sub('(?<=\w)\s*(?=[?.,!])', '', processed_text)

    # If there are multiple periods, replace with only one
    processed_text = re.sub('\.+', '.', processed_text)

    # If there are multiple periods, replace with only one
    processed_text = re.sub('\.+', '.', processed_text)

    # If there are comma followed by a period, replace with the former
    processed_text = re.sub('(?<=[^A-Za-z0-9\s])\.', '', processed_text)

    # If there is a full stop followed by a lower case character, then is probably part of the sentence
    processed_text = re.sub('\.\s*(?=[a-z])', ' ', processed_text)

    # Split by period, if there is only one character in the entry, fitler them out
    processed_text = '.'.join([i for i in processed_text.split('.') if len(i.split()) > 1])

    # Remove beginning white space if present. Adding period at the end
    processed_text = processed_text.strip() + '.'

    return processed_text


def text_length_less_than(text, length):
    """
    Function to check string length

    Prarmeters:
        text (str): Text of interest
        length (int): Minimum length of acceptable text

    Returns:
        True if text length is below length, otherwise False
    """
    text = final_cleaning(text)

    if len(re.findall(r'\w+', text)) < length:
        return True

    return False


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

    # Critera for text length,
    min_length = 100

    # Check if text length is smaller than min_length, if true return orginal text without any cleaning
    if text_length_less_than(text, min_length):
        verboseprinter(f'Text length below {min_length}. Return cleaned original text.')
        return final_cleaning(text)

    li_results = process_li_tag(text, nlp)
    p_list_results = process_p_list(text, nlp)
    p_results = process_p_tag(text, nlp)

    # After subsetting, we relax the min length criteria
    filt_min_length = 50

    if len(li_results) > 0:
        verboseprinter('List object detected')
        if text_length_less_than(li_results, filt_min_length):
            verboseprinter(f'Text length below {filt_min_length}. Return cleaned original text.')
            return final_cleaning(text)
        return final_cleaning(li_results)

    elif len(p_list_results) > 0:
        verboseprinter('Paragraph list detected')
        if text_length_less_than(p_list_results, filt_min_length):
            verboseprinter(f'Text length below {filt_min_length}. Return cleaned original text.')
            return final_cleaning(text)
        return final_cleaning(p_list_results)

    elif len(p_results) > 0:
        verboseprinter('Paragraphs detected')
        if text_length_less_than(p_results, filt_min_length):
            verboseprinter(f'Text length below {filt_min_length}. Return cleaned original text.')
            return final_cleaning(text)
        return final_cleaning(p_results)

    else:
        verboseprinter('None detected, returning all')
        return final_cleaning(re.sub('<.*?>', ' ', text))
