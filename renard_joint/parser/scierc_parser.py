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

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Constants
TRAIN_PATH = "../../data/sciERC/processed_data/json/train.json"
DEV_PATH = "../../data/sciERC/processed_data/json/dev.json"
TEST_PATH = "../../data/sciERC/processed_data/json/test.json"
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
        docs.append(json.loads(line))
    return docs


# Data checkers
def check_doc(document):
    """Check if the data in the document is consistent"""
    try:
        assert "doc_key" in document
    except AssertionError:
        print("The following document does not have an ID")
        print(document)
        return
    doc_id = document["doc_key"]
    try:
        assert "sentences" in document
    except AssertionError:
        print("Document '", doc_id, "' does not have key 'sentences'")
        return
    try:
        assert "ner" in document
    except AssertionError:
        print("Document '", doc_id, "' does not have key 'ner'")
        return
    try:
        assert "relations" in document
    except AssertionError:
        print("Document '", doc_id, "' does not have key 'relations'")
        return
    try:
        assert len(document["sentences"]) == len(document["ner"]) == len(document["relations"])
    except AssertionError:
        print("Document '", doc_id, "' does not have consistent number of sentences")
        return
    # Check entity consistency
    total_length = sum([len(sentence) for sentence in document["sentences"]])
    for i in range(len(document["ner"])):
        for entity in document["ner"][i]:
            try:
                assert len(entity) == 3
                assert isinstance(entity[2], str)
                assert 0 <= entity[0] <= entity[1] < total_length
            except AssertionError:
                print("Document '", doc_id, "', entity", entity, "is inconsistent")
    # Check relation consistency
    for i in range(len(document["relations"])):
        for relation in document["relations"][i]:
            try:
                assert len(relation) == 5
                assert isinstance(relation[4], str)
                assert 0 <= relation[0] <= relation[1] < total_length
                assert 0 <= relation[2] <= relation[3] < total_length
            except AssertionError:
                print("Document '", doc_id, "', relation", relation, "is inconsistent")


def check_docs(group):
    """Check if all the documents contained in the data group is consistent
    'group' is either "train", "dev", or "test"
    """
    docs = get_docs(group)
    for document in docs:
        check_doc(document)


def check_data():
    """Check if the everything in the dataset is consistent
    Refer to check_docs(group) and check_doc(document)
    """
    check_docs("train")
    check_docs("dev")
    check_docs("test")


# Describers
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


def get_text_length_doc(document):
    """Get the length of the document and the length of each sentence in it"""
    sentence_lengths = [len(sentence) for sentence in document["sentences"]]
    return sentence_lengths, sum(sentence_lengths)


def describe_text_length(group):
    """Show information about the length of documents in the dataset group"""
    sentence_lengths = []
    document_lengths = []
    docs = get_docs(group)
    for document in docs:
        sen_lens, doc_len = get_text_length_doc(document)
        sentence_lengths += sen_lens
        document_lengths.append(doc_len)
    describe_list(sentence_lengths, "sentence lengths of " + group)
    describe_list(document_lengths, "document lengths of " + group)


def count_type_doc(document, type_name):
    """Count the number of each class of a type in a document"""
    count = {}
    for i in range(len(document[type_name])):
        for element in document[type_name][i]:
            if element[-1] in count:
                count[element[-1]] += 1
            else:
                count[element[-1]] = 1
    return count


def describe_type(group, type_name, describe=True):
    """Describe the types of a property in the dataset"""
    docs = get_docs(group)
    count = {}
    for document in docs:
        cnt = count_type_doc(document, type_name)
        for key in cnt:
            if key not in count:
                count[key] = cnt[key]
            else:
                count[key] += cnt[key]
    if describe:
        print("Description of '", type_name, "' in", group)
        print("Total:", sum(count.values()))
        for key in count:
            print(key, ":", count[key])
        sns.barplot(list(count.values()), list(count.keys()))
        plt.show()
    # Return a map from entities to corresponding encoding numbers
    return dict(zip(["None"] + list(count.keys()), range(len(count) + 1)))


def get_sentence_number(sentence_lengths, position):
    """Given a list of sentence lengths, return the number to which the position belongs"""
    i = 0
    total = sentence_lengths[0]
    while position >= total:
        i += 1
        total += sentence_lengths[i]
    return i


def count_cross_sentence_relations_doc(document):
    "Count the number of cross sentence relations in a document"
    count = 0
    sentence_lengths = [len(sentence) for sentence in document["sentences"]]
    for i in range(len(document["relations"])):
        for relation in document["relations"][i]:
            try:
                assert get_sentence_number(sentence_lengths, relation[0]) == i
                assert get_sentence_number(sentence_lengths, relation[2]) == i
            except AssertionError:
                count += 1
    return count


def count_cross_sentence_relations(group):
    "Count the number of cross sentence relations in a dataset"
    count = 0
    docs = get_docs(group)
    for document in docs:
        count += count_cross_sentence_relations_doc(document)
    print("There are", count, "cross sentence relations group", group)


def describe_data():
    print("Description of train dataset:")
    describe_text_length("train")
    describe_type("train", "ner")
    describe_type("train", "relations")
    print("---------------------------------------------------------------------------------")
    print("Description of dev dataset:")
    describe_text_length("dev")
    describe_type("dev", "ner")
    describe_type("dev", "relations")
    print("---------------------------------------------------------------------------------")
    print("Description of test dataset:")
    describe_text_length("test")
    describe_type("test", "ner")
    describe_type("test", "relations")
    print("---------------------------------------------------------------------------------")
    count_cross_sentence_relations("train")
    count_cross_sentence_relations("dev")
    count_cross_sentence_relations("test")


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


# Checker
def check_extracted_data(data):
    """Check if all extracted data is valid"""
    for item in data:
        document_name = item["document"]
        data_frame = item["data_frame"]
        entities = item["entities"]
        relations = item["relations"]

        # Check if sentence embedding are correct
        sentence_embedding = data_frame["sentence_embedding"].tolist()
        for i in range(1, len(sentence_embedding)):
            try:
                assert 0 <= sentence_embedding[i] - sentence_embedding[i - 1] <= 1
            except AssertionError:
                print("Check failed at document", document_name, "in 'sentence_embedding' at position",
                      i, "sentence_embedding[i] - sentence_embedding[i-1] =",
                      sentence_embedding[i] - sentence_embedding[i - 1])

        # Check if all relations are valid
        for value in relations.values():
            first = value["source"]
            second = value["target"]
            try:
                assert first in entities
            except AssertionError:
                print("Check failed at document", document_name, "in 'relations',", first,
                      "is not found in record")
            try:
                assert second in entities
            except AssertionError:
                print("Check failed at document", document_name, "in 'relations',", second,
                      "is not found in record")
