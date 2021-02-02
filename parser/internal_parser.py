"""
USAGE:

Adapt the constants:
RECORD_PATH is the record file that contains the names(ids) of all documents,
DATA_PATH is the folder that contains the documents
UNK_TOKEN is the id of the unknown token (only for describing data)

Main functions:
check_data(): Check if the record and the documents are consistent
extract_data(): Parse and return the dataset
check_extracted_data(data): Check if the parsed data is consistent
describe_data(): Describe the dataset

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bisect

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Constants
RECORD_PATH = "C:\\Users\\NGUYEMI\\Work\\data\\documents.json"
DATA_PATH = "C:\\Users\\NGUYEMI\\Work\\data\\gt\\"
UNK_TOKEN = 100


# -- Functions ------------------------------------------------------------------------------------------------------- #
# Getters
def get_record():
    """Read the document record file"""
    record_file = open(RECORD_PATH, "r", encoding="utf8")
    record = json.load(record_file)
    return record


def get_doc(document_name):
    """Given a name, return the corresponding document"""
    return json.load(open(DATA_PATH + document_name + ".json", "r", encoding="utf8"))


# Data checkers
def check_record(record=get_record()):
    """Check if the number of documents in the record matches the number of data files"""
    assert len(record) == len(os.listdir(DATA_PATH))


def check_doc(document_name):
    """Check if extracted sentences and words have correct location tags
    Check if relations contain exactly 2 arguments
    """
    doc = get_doc(document_name)
    for sentence in doc["sentences"]:
        try:
            assert sentence["text"] == doc["text"][sentence["begin"]:sentence["end"]]
        except AssertionError:
            print("Error in doc",
                  document_name,
                  "sentence",
                  sentence["id"],
                  "'",
                  sentence["text"],
                  "' does not match '",
                  doc["text"][sentence["begin"]:sentence["end"]])
        for word in sentence["tokens"]:
            try:
                assert word["text"] == doc["text"][word["begin"]:word["end"]]
            except AssertionError:
                print("Error in doc",
                      document_name,
                      "sentence",
                      sentence["id"],
                      "word",
                      word["id"],
                      "'",
                      word["text"],
                      "' does not match '",
                      doc["text"][word["begin"]:word["end"]])
    for relation in doc["relations"]:
        try:
            assert len(relation["args"]) == 2
        except AssertionError:
            print("Error in doc",
                  document_name,
                  "relation",
                  relation["id"],
                  "has",
                  len(relation["args"]),
                  "arguments")


def check_data(record=get_record()):
    """Check if the dataset is in good shape
    Refer to check_record() and check_doc()
    """
    check_record(record)
    for document in record:
        check_doc(document["id"])


# Describers
def get_text_length_doc(document_name):
    """Get text length and sentence length of a document"""
    text_length = 0
    sentence_lengths = []
    doc = get_doc(document_name)
    for sentence in doc["sentences"]:
        text_length += len(sentence["tokens"])
        sentence_lengths.append(len(sentence["tokens"]))
    return text_length, sentence_lengths


def describe_list(lst, name):
    """Show the properties of the given sequence"""
    print("Description of", name, ":")
    print("Count:", len(lst))
    print("Sum:", sum(lst))
    print("Min:", min(lst))
    print("Mean:", sum(lst) / len(lst))
    print("Max:", max(lst))
    sns.distplot(lst, axlabel=name)
    plt.show()
    print()


def describe_text_length(record=get_record()):
    """Show information about the length of text and sentences in the dataset"""
    text_lengths = []
    sentence_lengths = []
    for document in record:
        txt_len, snt_len = get_text_length_doc(document["id"])
        text_lengths.append(txt_len)
        sentence_lengths += snt_len
    describe_list(text_lengths, "Text length")
    describe_list(sentence_lengths, "Sentence length")


def count_type_doc(document_name, type_name):
    """Count the number of each type of a property in a document"""
    doc = get_doc(document_name)
    count = {}
    for item in doc[type_name]:
        if item["type"] not in count:
            count[item["type"]] = 1
        else:
            count[item["type"]] += 1
    return count


def describe_type(type_name, record=get_record(), describe=True):
    """Describe the types of a property in the dataset"""
    count = {}
    for document in record:
        cnt = count_type_doc(document["id"], type_name)
        for key in cnt:
            if key not in count:
                count[key] = cnt[key]
            else:
                count[key] += cnt[key]
    if describe:
        print("Description of", type_name)
        print("Total:", sum(count.values()))
        for key in count:
            print(key, ":", count[key])
        sns.barplot(list(count.values()), list(count.keys()))
        plt.show()
    # Return a map from entities to corresponding encoding numbers
    return dict(zip(["None"] + list(count.keys()), range(len(count) + 1)))


# Parsers
entity_encode = describe_type("mentions", describe=False)
relation_encode = describe_type("relations", describe=False)


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
            # Put words to lower case to use uncased bert model
            words.append(word["text"].lower())
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
    try:
        assert len(token_id) == len(words) == len(begins) == len(ends) == len(sentence_embedding)
    except AssertionError:
        print("Input lists do not have the same length, abort")
        return token_id, words, begins, ends, sentence_embedding
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


def get_relation_doc(document_name, entity_position):
    """Extract relations from a document"""
    doc = get_doc(document_name)
    relation_position = {}
    for relation in doc["relations"]:
        relation_position[relation["id"]] = tuple(relation["args"])
    return relation_position


def extract_doc(document_name):
    """Extract data from a document to a pandas dataset"""
    data_frame = pd.DataFrame()
    words, begins, ends, sentence_embedding = get_word_doc(document_name)
    token_ids = get_token_id(words)
    data_frame["token_ids"], data_frame["words"], data_frame["begins"], data_frame["ends"], \
        data_frame["sentence_embedding"] = expand_token_id(token_ids, words, begins, ends, sentence_embedding)
    data_frame["tokens"] = tokenizer.convert_ids_to_tokens(data_frame["token_ids"])
    entity_position, data_frame["entity_embedding"] = get_entity_doc(document_name, list(data_frame["begins"]))
    relation_position = get_relation_doc(document_name, entity_position)
    return {"document": document_name,
            "data_frame": data_frame,
            "entity_position": entity_position,
            "relation_position": relation_position}


def extract_data(record=get_record()):
    """Extract all documents to a dataset for training"""
    data = []
    for document in record:
        data.append(extract_doc(document["id"]))
    return data


# Checkers
def check_extracted_data(data):
    """
    .. todo: too complex (26)
    Check if all extracted data is valid"""
    for item in data:
        document_name = item["document"]
        data_frame = item["data_frame"]
        entity_position = item["entity_position"]
        relation_position = item["relation_position"]

        # Check if begins is increasing
        begins = data_frame["begins"].tolist()
        for i in range(1, len(begins)):
            try:
                assert begins[i] >= begins[i - 1]
            except AssertionError:
                print("Check failed at document", document_name, "in 'begins' at position", i - 1,
                      "(value", begins[i - 1], ") >", i, "(value", begins[i], ")")

        # Check if ends is increasing
        ends = data_frame["ends"].tolist()
        for i in range(1, len(ends)):
            try:
                assert ends[i] >= ends[i - 1]
            except AssertionError:
                print("Check failed at document", document_name, "in 'ends' at position", i - 1,
                      "(value", ends[i - 1], ") >", i, "(value", ends[i], ")")

        # Check if ends are always greater than begins
        for i in range(len(begins)):
            try:
                assert begins[i] < ends[i]
            except AssertionError:
                print("Check failed at document", document_name, "in 'begins' & 'ends' at position",
                      i, "(begin", begins[i], ">= end", ends[i], ")")

        # Check if sentence embedding are correct
        sentence_embedding = data_frame["sentence_embedding"].tolist()
        for i in range(1, len(sentence_embedding)):
            try:
                assert 0 <= sentence_embedding[i] - sentence_embedding[i - 1] <= 1
            except AssertionError:
                print("Check failed at document", document_name, "in 'sentence_embedding' at position",
                      i, "sentence_embedding[i] - sentence_embedding[i-1] =",
                      sentence_embedding[i] - sentence_embedding[i - 1])
        try:
            assert sentence_embedding[-1] == len(get_doc(document_name)["sentences"]) - 1
        except AssertionError:
            print("Check failed at document", document_name, ", expected",
                  len(get_doc(document_name)["sentences"]), "sentences but", sentence_embedding[-1] + 1, "found")

        # Check if entities correctly embedded
        entity_embedding = data_frame["entity_embedding"].tolist()
        cnt = 0
        for entity_key in entity_position:
            low, high = entity_position[entity_key]
            cnt += high - low
            try:
                assert min(entity_embedding[low:high]) == max(entity_embedding[low:high])
            except AssertionError:
                print("Check failed at document", document_name, "in 'entity_embedding', key", entity_key,
                      ", values from", low, "to", high, ":", entity_embedding[low:high], "are inconsistent")
        try:
            assert cnt == (np.array(entity_embedding) != 0).astype(int).sum()
        except AssertionError:
            print("Check failed at document", document_name, "in total entity embedded tokens",
                  (np.array(entity_embedding) != 0).astype(int).sum(), "does not match the record", cnt)

        # Check if all relations are valid
        for first, second in relation_position.values():
            try:
                assert first in entity_position
            except AssertionError:
                print("Check failed at document", document_name, "in 'relation_position',", first,
                      "is not found in record")
            try:
                assert second in entity_position
            except AssertionError:
                print("Check failed at document", document_name, "in 'relation_position',", second,
                      "is not found in record")


# Describers
def describe_token(data):
    """Describe the tokens & entities in the dataset"""
    token_count = 0
    entity_token_count = 0
    unknown_token_count = 0
    unknown_entity_token_count = 0
    for item in data:
        data_frame = item["data_frame"]
        token_count += data_frame["token_ids"].count()
        entity_token_count += data_frame[data_frame["entity_embedding"] > 0]["token_ids"].count()
        unknown_token_count += data_frame[data_frame["token_ids"] == UNK_TOKEN]["token_ids"].count()
        unknown_entity_token_count += data_frame[(data_frame["entity_embedding"] > 0) &
                                                 (data_frame["token_ids"] == UNK_TOKEN)]["token_ids"].count()
    print("Token count:", token_count)
    print("Entity token count:", entity_token_count)
    print("Unknown token count:", unknown_token_count)
    print("Unknown entity token count:", unknown_entity_token_count)


def describe_relation(data):
    """Describe the relation in the dataset"""
    relation_count = 0
    reverse_relation_count = 0
    cross_sentence = []
    for item in data:
        sentence_embedding = item["data_frame"]["sentence_embedding"].tolist()
        entity_position = item["entity_position"]
        relation_position = item["relation_position"]
        for first, second in relation_position.values():
            relation_count += 1
            if entity_position[first][0] > entity_position[second][0]:
                reverse_relation_count += 1
            if sentence_embedding[entity_position[first][0]] != sentence_embedding[entity_position[second][0]]:
                cross_sentence.append(abs(sentence_embedding[entity_position[first][0]] -
                                          sentence_embedding[entity_position[second][0]]))
    print("Relation count:", relation_count)
    print("Reverse relation count:", reverse_relation_count)
    print("Cross sentence count:", sum(cross_sentence))
    # sns.countplot(cross_sentence)
    plt.show()


def describe_data(record=get_record()):
    """Show information about the dataset
    Refer to describe_text_length(), describe_type(), and describe_token()
    """
    describe_text_length(record)
    print()

    entity_encode = describe_type("mentions", record)
    print("Entity encoding:", entity_encode)
    print()

    relation_encode = describe_type("relations", record)
    print("Relation encoding:", relation_encode)
    print()

    data = extract_data(record)
    describe_token(data)
    print()
    describe_relation(data)
    print()
