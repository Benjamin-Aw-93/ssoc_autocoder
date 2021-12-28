import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import time
import nltk
import math
import random
import regex as re
from .utils import verboseprint

# Loading in the necessary file to run augmnetation
nltk.download('averaged_perceptron_tagger')

# Load verbosity ideally should load in command line, write as -v tag in cmd
# Should load load at the start of the script
verbosity = False  # default value

verboseprinter = verboseprint(verbosity)


# Function to trim text due to model constraints
def trim(text, max_len):

    arr_text = text.split()

    if len(arr_text) > max_len:

        verboseprinter(f"Text is too long with length {len(arr_text)}. Trimmed.")

        idx_lower = math.ceil((len(arr_text) - max_len) / 2)

        idx_upper = idx_lower + max_len

        text = " ".join(arr_text[idx_lower:idx_upper])

    return text


# Initialisating prarmeters for augmentation
params = {
    'top_k': 5,
    "top_p": 0.9,
    'aug_p': 0.5,
    'aug_min': 10,
    'device': 'cpu',
    'min_length': 50,
    'wrd_embd': {
        'model_type': 'glove',
        'model_path': '../Models/glove.840B.300d.txt',
        'action': 'substitute'
    },
    'bk_trans': {
        'from_model_name': 'facebook/wmt19-en-de',
        'to_model_name': 'facebook/wmt19-de-en',
    },
    'synonym': {
        'aug_src': 'ppdb',
        'model_path': '../Models/ppdb-2.0-tldr'
    },
    'context_emb': {
        'model_path': 'distilbert-base-uncased',
        'action': 'substitute'
    },
    'sent_aug': {
        'model_path': 'distilgpt2'
    },
    'summ_aug': {
        'model_path': 't5-base'
    }
}


def data_aug_collated(text, params):
    """
    Augmentation of a particular text

    Parameters:
        text (str): Text to be augmented
        params (dict): Pre-defined parameters

    Returns:
        A dictionary of augmentated outputs
    """

    start = time.perf_counter()
    output = {'orginal_text': text}

    # Context embedding and translation will break if max length is not handled properly
    text_length_max = int(len(text.split()) * 1.25)

    verboseprinter(f"Original text:\n{text}")
    verboseprinter(
        "===================================================================================================")

    tic = time.perf_counter()
    wrd_embd_aug = naw.WordEmbsAug(model_type=params['wrd_embd']['model_type'],
                                   model_path=params['wrd_embd']['model_path'],
                                   action=params['wrd_embd']['action'],
                                   top_k=params['top_k'],
                                   aug_p=params['aug_p'],
                                   aug_min=params['aug_min'])

    wrd_emb_out = wrd_embd_aug.augment(text, num_thread=4)
    output["wrd_emb_out"] = wrd_emb_out.strip()
    toc = time.perf_counter()

    verboseprinter(f"Word embedding convertion:\n{wrd_emb_out}")
    verboseprinter(f"\nTime taken: {toc - tic:0.4f}")
    verboseprinter(
        "===================================================================================================")

    tic = time.perf_counter()
    bk_trans_aug = naw.BackTranslationAug(from_model_name=params['bk_trans']['from_model_name'],
                                          to_model_name=params['bk_trans']['to_model_name'],
                                          device=params['device'],
                                          max_length=text_length_max)

    bk_trans_out = bk_trans_aug.augment(trim(text, 200), num_thread=4)
    output["bk_trans_out"] = bk_trans_out.strip()
    toc = time.perf_counter()

    verboseprinter(f"Back translation convertion:\n{bk_trans_out}")
    verboseprinter(f"\nTime taken: {toc - tic:0.4f}")
    verboseprinter(
        "===================================================================================================")

    tic = time.perf_counter()
    synonym_aug = naw.SynonymAug(aug_src=params['synonym']['aug_src'],
                                 model_path=params['synonym']['model_path'],
                                 aug_p=params['aug_p'],
                                 aug_min=params['aug_min'])

    synonym_out = synonym_aug.augment(text, num_thread=4)
    output["synonym_out"] = synonym_out.strip()
    toc = time.perf_counter()

    verboseprinter(f"Synonym convertion:\n{synonym_out}")
    verboseprinter(f"\nTime taken: {toc - tic:0.4f}")
    verboseprinter(
        "===================================================================================================")

    tic = time.perf_counter()

    context_emb_aug = naw.ContextualWordEmbsAug(model_path=params['context_emb']['model_path'],
                                                action=params['context_emb']['action'],
                                                top_k=params['top_k'],
                                                aug_p=params['aug_p'],
                                                aug_min=params['aug_p'],
                                                device=params['device'])

    context_emb_out = context_emb_aug.augment(text, num_thread=4)
    output["context_emb_out"] = context_emb_out.strip()
    toc = time.perf_counter()

    verboseprinter(f"Context embedding convertion:\n{context_emb_out}")
    verboseprinter(f"\nTime taken: {toc - tic:0.4f}")
    verboseprinter(
        "===================================================================================================")

    sent_aug = nas.ContextualWordEmbsForSentenceAug(model_path=params['sent_aug']['model_path'],
                                                    min_length=params['min_length'],
                                                    max_length=text_length_max,
                                                    top_k=params['top_k'],
                                                    top_p=params['top_p'],
                                                    device=params['device'])

    sent_out = sent_aug.augment(trim(text, 500), num_thread=4)
    output["sent_out"] = sent_out.strip()
    toc = time.perf_counter()

    verboseprinter(f"Sentence augmentation convertion:\n{sent_out}")
    verboseprinter(f"\nTime taken: {toc - tic:0.4f}")
    verboseprinter(
        "===================================================================================================")

    summ_aug = nas.AbstSummAug(model_path=params['summ_aug']['model_path'],
                               min_length=params['min_length'],
                               top_k=params['top_k'])

    summ_out = summ_aug.augment(trim(text, 400), num_thread=4)

    verboseprinter(f"Summarisation convertion:\n{summ_out}")
    output["summ_out"] = summ_out.strip()
    verboseprinter(f"\nTime taken: {toc - tic:0.4f}")
    verboseprinter(
        "===================================================================================================")

    end = time.perf_counter()
    verboseprinter("Done convertion")
    verboseprinter(f"\nTotal time taken: {end - start:0.4f}")

    return output


# Random injection of common phrases, inserted as noise
common_phrases = [' We are looking for/searching for a candidate who is',
                  ' We are looking for a candidate who can',
                  ' Are you passionate about this job',
                  ' Do you love a job that',
                  ' We are a company that',
                  ' We are a startup that',
                  ' We are a agency that',
                  ' Would you like to work for a company that is',
                  ' Are you interested in working for a company that',
                  ' In this role, you will be responsible for',
                  ' One of your key responsibilities in this job will be',
                  ' If you love this role, then you’ll fit right in our team of',
                  ' If you would like to be part of our team, apply today by',
                  ' Sound like you? Then, send your resumé/CV and cover letter to',
                  ' If this sounds like you, then apply by clicking the button below',
                  ' Description of the duties and responsibilities of the job includes',
                  ' This job is ideal for someone who is',
                  ' Top skills and proficiencies include',
                  ' Dynamic work environment',
                  ' Proven track record',
                  ' Must be a self-starter',
                  ' Detailed orientated',
                  ' Requires to have good communication skills',
                  ' We are looking for a candidate who works independently']


def random_insert(text, common_phrases, prob, edit_phrase):
    """
    Insert a phrases randomly into the text

    Parameters:
        text (str): Text to check for
        common_phrases (list): Pre-defined parameters
        prob (float): probability of augmenting a particular sentence
        edit_phrase (bool): Whether to edit common phrases using distilbert

    Returns:
        A dictionary of augmentated outputs
    """

    text_list = text.split(".")
    length = len(text_list)
    selected_phrases = random.sample(common_phrases, math.ceil(len(common_phrases) * prob))

    if edit_phrase:
        selected_phrases = [naw.ContextualWordEmbsAug(
            model_path='distilbert-base-uncased', action="substitute").augment(phrase, num_thread=4) for phrase in selected_phrases]

        selected_phrases = [
            " " + " ".join(re.sub('[^a-zA-Z ]+', '', phrase).capitalize().split()) for phrase in selected_phrases]

    verboseprinter(f"Phrases that were added in: {selected_phrases}")

    for phrase in selected_phrases:
        text_list.insert(random.randrange(length), phrase)
        length = len(text_list)

    return '.'.join(text_list)


def data_augmentation(text, prob, edit_phrase, params=params, common_phrases=common_phrases):
    """
    Combining random insert with augmentation outputs to create final textual outputs

    Parameters:
        text (str): Text to check for
        params (dict): Pre-defined parameters
        common_phrases (list): Pre-defined parameters
        prob (float): probability of augmenting a particular sentence
        edit_phrase (bool): Whether to edit common phrases using distilbert

    Returns:
        A dictionary of augmentated outputs
    """

    output = data_aug_collated(text, params)

    output = {augment: random_insert(text, common_phrases, prob, edit_phrase)
              for augment, text in output.items()}

    return output
