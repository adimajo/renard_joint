"""
USAGE:

1. Adapt the constants:
TRAIN_PATH is the path to the train dataset,
DEV_PATH is the path to the dev dataset,
TEST_PATH is the path to the test dataset,

2. Obtain the list of documents from one of three group: "train", "dev", or "test"
docs = get_docs(group)

3. Main functions:
check_data(): Check if the record and the documents are consistent
data = extract_data(group): Parse and return the dataset
check_extracted_data(data): Check if the parsed data is consistent
describe_data(): Describe the whole dataset

"""
import os
import pandas as pd
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Constants
TRAIN_PATH = os.environ["DATA"] + "CoNLL04/train.txt"
DEV_PATH = os.environ["DATA"] + "CoNLL04/dev.txt"
TEST_PATH = os.environ["DATA"] + "CoNLL04/test.txt"
UNK_TOKEN = 100
CLS_TOKEN = 101
SEP_TOKEN = 102

entity_encode = {'O': 0, 'B-Loc': 1, 'I-Loc': 2, 'B-Peop': 3, 'I-Peop': 4,
                 'B-Org': 5, 'I-Org': 6, 'B-Other': 7, 'I-Other': 8}
relation_encode = {'N': 0, 'Kill': 1, 'Located_In': 2, 'OrgBased_In': 3,
                   'Live_In': 4, 'Work_For': 5}


# -- Functions ------------------------------------------------------------------------------------------------------- #
# Getters
def get_docs(group):
    """Read the dataset group and return a list of documents
    'group' is either "train", "dev", or "test"
    """
    if group == "train":
        dataset = open(TRAIN_PATH, "r", encoding="utf8").readlines()
    elif group == "dev":
        dataset = open(DEV_PATH, "r", encoding="utf8").readlines()
    elif group == "test":
        dataset = open(TEST_PATH, "r", encoding="utf8").readlines()
    else:
        print("No '", group, "' data group found!")
        return []
    docs = []
    for line in dataset:
        if line.startswith("#"):
            docs.append([])
        docs[-1].append(line)
    return docs


def get_text_length_doc(document):
    """Get the length of a document"""
    return len(document) - 1


def split_line(line):
    """Split a line in the following format from the dataset
    index word entity_type ['relation_type_1', ...] ['target_entity_1', ...]
    """
    index, word, entity_type, rest = tuple(line.split(maxsplit=3))
    rt, te = tuple(rest.split("]", 1))
    relation_types = [item.strip("\r\n\t []'") for item in rt.split(",")]
    target_entities = [int(item.strip("\r\n\t []'")) for item in te.split(",")]
    assert len(relation_types) == len(target_entities), "Relation types and target entities not same length"
    return index, word, entity_type, relation_types, target_entities


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


def expand_token_id(ids, token_ids, words, entity_embedding, entity_position):
    """Expand token id and duplicate members in other list wherever necessary"""
    # Test if all lists have the same length as expected
    assert len(ids) == len(token_ids) == len(words) == len(entity_embedding), "Input lists do not have" \
                                                                              " the same length, abort"
    new_ids = []
    new_token_ids = []
    new_words = []
    new_entity_embedding = []
    id_range = {}
    last = 0
    for i in range(len(ids)):
        for tid in token_ids[i]:
            new_ids.append(ids[i])
            new_token_ids.append(tid)
            new_words.append(words[i])
            new_entity_embedding.append(entity_embedding[i])
        id_range[ids[i]] = (last, len(new_ids))
        last = len(new_ids)
    for key in entity_position:
        entity_position[key] = (id_range[entity_position[key][0]][0],
                                id_range[entity_position[key][1] - 1][1])
    return new_ids, new_token_ids, new_words, new_entity_embedding, entity_position


def extract_doc(document):
    """Extract data from a document"""
    doc_id = document[0].split()[1]
    index = []
    words = []
    entity_embedding = []
    entity_id = []
    relation_embedding = []
    target_entity_embedding = []
    entity_position = {}
    relations = {}
    entity_count = 0
    relation_count = 0
    # Parse the document
    for line in document[1:]:
        idx, word, entity_type, relation_types, target_entities = split_line(line)
        index.append(int(idx))
        words.append(word)
        entity_embedding.append(entity_encode[entity_type])
        relation_embedding.append(relation_types)
        target_entity_embedding.append(target_entities)
        # if an I appears after an O, assume it's the start of a new entity
        if entity_type.startswith("B") or \
                (entity_type.startswith("I") and len(entity_embedding) >= 2 and entity_embedding[-2] == 0):
            entity_count += 1
            entity_id.append(entity_count)
            entity_position[entity_count] = (int(idx), int(idx) + 1)
        elif entity_type.startswith("I"):
            entity_id.append(entity_count)
            entity_position[entity_count] = (entity_position[entity_count][0], int(idx) + 1)
        else:
            entity_id.append(0)
    # Parse the relations
    for idx in index:
        if "N" not in relation_embedding[int(idx)]:
            for relation, target in zip(relation_embedding[int(idx)], target_entity_embedding[int(idx)]):
                relation_count += 1
                relations[relation_count] = {"type": relation_encode[relation],
                                             "source": entity_id[int(idx)],
                                             "target": entity_id[target]}
    # Tokenize and expand
    token_ids = get_token_id(words)
    data_frame = pd.DataFrame()
    data_frame["ids"], data_frame["token_ids"], data_frame["words"], data_frame["entity_embedding"], \
        entity_position = expand_token_id(index, token_ids, words, entity_embedding, entity_position)
    data_frame["tokens"] = tokenizer.convert_ids_to_tokens(data_frame["token_ids"])
    return {"document": doc_id,
            "data_frame": data_frame,
            "entity_position": entity_position,
            "relations": relations}


def extract_data(group):
    """Extract all documents to a dataset"""
    docs = get_docs(group)
    data = []
    for document in docs:
        data.append(extract_doc(document))
    return data
