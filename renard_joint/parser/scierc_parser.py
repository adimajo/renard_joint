"""
Parser for SciERC data.

.. autosummary::
    :toctree:

    get_docs
    get_token_id
    expand_token_id
    extract_doc
    extract_data
"""
import os
import json

import pandas as pd
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Constants
TRAIN_PATH = os.path.join(os.environ["DATA"], "sciERC/train.json")
DEV_PATH = os.path.join(os.environ["DATA"], "sciERC/dev.json")
TEST_PATH = os.path.join(os.environ["DATA"], "sciERC/test.json")
UNK_TOKEN = 100
CLS_TOKEN = 101
SEP_TOKEN = 102

entity_encode = {'None': 0, 'Task': 1, 'OtherScientificTerm': 2, 'Material': 3, 'Generic': 4, 'Method': 5,
                 'Metric': 6}
relation_encode = {'None': 0, 'PART-OF': 1, 'USED-FOR': 2, 'HYPONYM-OF': 3, 'CONJUNCTION': 4, 'FEATURE-OF': 5,
                   'EVALUATE-FOR': 6, 'COMPARE': 7}


# -- Functions ------------------------------------------------------------------------------------------------------- #
# Getters
def get_docs(group):
    """Read the dataset group and return a list of documents
    'group' is either "train", "dev", or "test"
    """
    if group.lower() == "train":
        dataset = open(TRAIN_PATH, "r", encoding="utf8").readlines()
    elif group.lower() == "dev":
        dataset = open(DEV_PATH, "r", encoding="utf8").readlines()
    elif group.lower() == "test":
        dataset = open(TEST_PATH, "r", encoding="utf8").readlines()
    else:
        print("No '", group, "' data group found!")
        return []
    docs = []
    for line in dataset:
        docs.append(json.loads(line))
    return docs


# Parsers
def get_token_id(words):
    """Tokenize each word in a list of words
    Return a list of lists of token ids
    """
    token_id = []
    for word in words:
        # apply [1:-1] to remore CLS and SEP token ids at the begin and the end of the list
        token_id.append(tokenizer(word)["input_ids"][1:-1])
    return token_id


def expand_token_id(token_ids, words, sentence_embedding, entities):
    """Expand token id and duplicate members in other lists wherever necessary"""
    # Test if all lists have the same length as expected
    try:
        assert len(token_ids) == len(words) == len(sentence_embedding)
    except AssertionError:
        print("Input lists do not have the same length, abort")
        return token_ids, words, sentence_embedding, entities
    new_token_ids = []
    new_words = []
    new_sentence_embedding = []
    id_range = {}
    last = 0
    for i in range(len(token_ids)):
        for tid in token_ids[i]:
            new_token_ids.append(tid)
            new_words.append(words[i])
            new_sentence_embedding.append(sentence_embedding[i])
        id_range[i] = (last, len(new_token_ids))
        last = len(new_token_ids)
    for key in entities:
        entities[key]["begin"] = id_range[entities[key]["begin"]][0]
        entities[key]["end"] = id_range[entities[key]["end"] - 1][1]
    return new_token_ids, new_words, new_sentence_embedding, entities


def extract_doc(document):
    """Extract data from a document"""
    doc_id = document["doc_key"]
    words = []
    sentence_embedding = []
    entities = {}
    entity_span = {}
    relations = {}
    entity_count = 0
    relation_count = 0
    # Parse the words
    for i in range(len(document["sentences"])):
        for word in document["sentences"][i]:
            words.append(word)
            sentence_embedding.append(i)
    # Parse the entities
    for i in range(len(document["ner"])):
        for entity in document["ner"][i]:
            entity_count += 1
            entities[entity_count] = {"type": entity_encode[entity[2]],
                                      "begin": entity[0], "end": entity[1] + 1}
            entity_span[(entity[0], entity[1] + 1)] = entity_count
    # Parse the relations
    for i in range(len(document["relations"])):
        for relation in document["relations"][i]:
            relation_count += 1
            relations[relation_count] = {"type": relation_encode[relation[4]],
                                         "source": entity_span[(relation[0], relation[1] + 1)],
                                         "target": entity_span[(relation[2], relation[3] + 1)]}
    # Tokenize and expand
    token_ids = get_token_id(words)
    data_frame = pd.DataFrame()
    data_frame["token_ids"], data_frame["words"], data_frame["sentence_embedding"], entities = \
        expand_token_id(token_ids, words, sentence_embedding, entities)
    data_frame["tokens"] = tokenizer.convert_ids_to_tokens(data_frame["token_ids"])
    return {"document": doc_id,
            "data_frame": data_frame,
            "entities": entities,
            "relations": relations}


def extract_data(group):
    """Extract all documents to a dataset"""
    docs = get_docs(group)
    data = []
    for document in docs:
        data.append(extract_doc(document))
    return data
