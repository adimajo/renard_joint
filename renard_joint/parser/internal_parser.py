"""
USAGE:

1. Adapt the constants:
RECORD_PATH is the record file that contains the names(ids) of all documents,
DATA_PATH is the folder that contains the documents

2. Obtain the list of documents
docs = get_docs("All", "Training", or "Test")

3. Main functions:
check_data(docs): Check if the record and the documents are consistent
data = extract_data(docs): Parse and return the dataset
check_extracted_data(data): Check if the parsed data is consistent
describe_data(docs): Describe the dataset

"""
import os
import bisect
import json

import pandas as pd
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Constants
RECORD_PATH = os.environ["DATA"] + "/internal_data/sets.json"
DATA_PATH = os.environ["DATA"] + "/internal_data/gt/"
UNK_TOKEN = 100
CLS_TOKEN = 101
SEP_TOKEN = 102

entity_encode = {'None': 0,
                 'EnvironmentalIssues': 1,
                 'Date': 2,
                 'Organisation': 3,
                 'CommitmentLevel': 4,
                 'Location': 5,
                 'CoalActivity': 6,
                 'SocialIssues': 7,
                 'SocialOfficialTexts': 8}
relation_encode = {'None': 0,
                   'Makes': 1,
                   'Of': 2,
                   'IsRelatedTo': 3,
                   'HasActivity': 4,
                   'Recognizes': 5,
                   'In': 6,
                   'IsInvolvedIn': 7}
# These encodings can be obtained automatically instead of hard-coding by running describe_type(), see examples below


# -- Functions ------------------------------------------------------------------------------------------------------- #
# Getters
def get_docs(group="All"):
    """Read the record and return a list of document names
    'group' is either "All", "Training", or "Test"
    """
    record_file = open(RECORD_PATH, "r", encoding="utf8")
    record = json.load(record_file)
    for docs in record:
        if docs["name"] == group:
            return docs["documents"]
    print("No '", group, "' data group found!")
    return []


def get_doc(document_name):
    """Given a name, return the corresponding document"""
    return json.load(open(DATA_PATH + document_name + ".json", "r", encoding="utf8"))


def get_word_doc(document_name):
    """Extract words and their position from a document"""
    doc = get_doc(document_name)
    words = []
    begins = []
    ends = []
    sentence_embedding = []
    sentence_count = 0
    for sentence in doc["sentences"]:
        for word in sentence["tokens"]:
            words.append(word["text"])
            begins.append(word["begin"])
            ends.append(word["end"])
            sentence_embedding.append(sentence_count)
        sentence_count += 1
    return words, begins, ends, sentence_embedding


def get_token_id(words):
    """Tokenize each word in a list of words
    Return a list of lists of token ids
    """
    token_id = []
    for word in words:
        # apply [1:-1] to remore CLS and SEP token ids at the begin and the end of the list
        token_id.append(tokenizer(word)["input_ids"][1:-1])
    return token_id


def expand_token_id(token_id, words, begins, ends, sentence_embedding):
    """Expand token id and duplicate members in other list wherever necessary"""
    # Test if all lists have the same length as expected
    assert len(token_id) == len(words) == len(begins) == len(ends) == len(sentence_embedding), "Input lists do " \
                                                                                                   "not have the same" \
                                                                                                   " length, abort"
    new_token_id = []
    new_words = []
    new_begins = []
    new_ends = []
    new_sentence_embedding = []
    for i in range(len(token_id)):
        for tid in token_id[i]:
            new_token_id.append(tid)
            new_words.append(words[i])
            new_begins.append(begins[i])
            new_ends.append(ends[i])
            new_sentence_embedding.append(sentence_embedding[i])
    return new_token_id, new_words, new_begins, new_ends, new_sentence_embedding


def get_entity_doc(document_name, begins):
    """Extract entities from a document (only use AFTER token id has been expanded)"""
    doc = get_doc(document_name)
    entity_embedding = [0] * len(begins)
    entity_position = {}
    for mention in doc["mentions"]:
        low = bisect.bisect_left(begins, mention["begin"])
        high = bisect.bisect_left(begins, mention["end"])
        entity_position[mention["id"]] = (low, high)
        for i in range(low, high):
            entity_embedding[i] = entity_encode[mention["type"]]
    return entity_position, entity_embedding


def get_relation_doc(document_name):
    """Extract relations from a document"""
    doc = get_doc(document_name)
    relations = {}
    for relation in doc["relations"]:
        relations[relation["id"]] = {"type": relation_encode[relation["type"]],
                                     "source": relation["args"][0],
                                     "target": relation["args"][1]}
    return relations


def extract_doc(document_name):
    """Extract data from a document"""
    data_frame = pd.DataFrame()
    words, begins, ends, sentence_embedding = get_word_doc(document_name)
    token_ids = get_token_id(words)
    data_frame["token_ids"], data_frame["words"], data_frame["begins"], data_frame["ends"], \
        data_frame["sentence_embedding"] = expand_token_id(token_ids, words, begins, ends, sentence_embedding)
    data_frame["tokens"] = tokenizer.convert_ids_to_tokens(data_frame["token_ids"])
    entity_position, data_frame["entity_embedding"] = get_entity_doc(document_name, list(data_frame["begins"]))
    relations = get_relation_doc(document_name)
    return {"document": document_name,
            "data_frame": data_frame,
            "entity_position": entity_position,
            "relations": relations}


def extract_data(docs):
    """Extract all documents to a dataset for training"""
    data = []
    for document in docs:
        data.append(extract_doc(document))
    return data
