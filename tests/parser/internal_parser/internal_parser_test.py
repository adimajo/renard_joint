import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import tikzplotlib

from renard_joint.PathHandler import MyPathHandler, PathOverWrite
from renard_joint.parser import internal_parser

VALUE = "(value"

CHECK_FAILED_AT_DOCUMENT = "Check failed at document"

ERROR_IN_DOC = "Error in doc"


def check_docs(docs):
    """Check if the number of documents in the "All" group in the record matches the number of data files
    """
    assert len(docs) == len(os.listdir(MyPathHandler().get_path()))


def check_doc(document_name):
    """Check if extracted sentences and words have correct location tags
    Check if relations contain exactly 2 arguments
    """
    doc = internal_parser.get_doc(document_name)
    for sentence in doc["sentences"]:
        try:
            assert sentence["text"] == doc["text"][sentence["begin"]:sentence["end"]]
        except AssertionError:
            print(internal_parser.ERROR_IN_DOC,
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
                print(internal_parser.ERROR_IN_DOC,
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
            print(internal_parser.ERROR_IN_DOC,
                  document_name,
                  "relation",
                  relation["id"],
                  "has",
                  len(relation["args"]),
                  "arguments")


def check_data(docs):
    """Check if the dataset is in good shape
    Refer to check_docs() and check_doc()
    """
    check_docs(docs)
    for document in docs:
        check_doc(document)


def get_text_length_doc(document_name):
    """Get text length and sentence length of a document"""
    text_length = 0
    sentence_lengths = []
    doc = internal_parser.get_doc(document_name)
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
    tikzplotlib.save(name + ".tex")
    plt.show()
    print()


def describe_text_length(docs):
    """Show information about the length of text and sentences in the dataset"""
    text_lengths = []
    sentence_lengths = []
    for document in docs:
        txt_len, snt_len = get_text_length_doc(document)
        text_lengths.append(txt_len)
        sentence_lengths += snt_len
    describe_list(text_lengths, "Text length")
    describe_list(sentence_lengths, "Sentence length")


def count_type_doc(document_name, type_name):
    """Count the number of each type of a property in a document"""
    doc = internal_parser.get_doc(document_name)
    count = {}
    for item in doc[type_name]:
        if item["type"] not in count:
            count[item["type"]] = 1
        else:
            count[item["type"]] += 1
    return count


def describe_type(type_name, docs, describe=True):
    """Describe the types of a property in the dataset"""
    count = {}
    for document in docs:
        cnt = count_type_doc(document, type_name)
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
        tikzplotlib.save(type_name + ".tex")
        plt.show()
    # Return a map from entities to corresponding encoding numbers
    return dict(zip(["None"] + list(count.keys()), range(len(count) + 1)))


def check_extracted_data(data):
    """
    .. todo: too complex (26)
    Check if all extracted data is valid"""
    for item in data:
        document_name = item["document"]
        data_frame = item["data_frame"]
        entity_position = item["entity_position"]
        relations = item["relations"]

        # Check if begins is increasing
        begins = data_frame["begins"].tolist()
        for i in range(1, len(begins)):
            try:
                assert begins[i] >= begins[i - 1]
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'begins' at position", i - 1,
                      VALUE, begins[i - 1], ") >", i, VALUE, begins[i], ")")

        # Check if ends is increasing
        ends = data_frame["ends"].tolist()
        for i in range(1, len(ends)):
            try:
                assert ends[i] >= ends[i - 1]
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'ends' at position", i - 1,
                      VALUE, ends[i - 1], ") >", i, VALUE, ends[i], ")")

        # Check if ends are always greater than begins
        for i in range(len(begins)):
            try:
                assert begins[i] < ends[i]
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'begins' & 'ends' at position",
                      i, "(begin", begins[i], ">= end", ends[i], ")")

        # Check if sentence embedding are correct
        sentence_embedding = data_frame["sentence_embedding"].tolist()
        for i in range(1, len(sentence_embedding)):
            try:
                assert 0 <= sentence_embedding[i] - sentence_embedding[i - 1] <= 1
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'sentence_embedding' at position",
                      i, "sentence_embedding[i] - sentence_embedding[i-1] =",
                      sentence_embedding[i] - sentence_embedding[i - 1])
        try:
            assert sentence_embedding[-1] == len(internal_parser.get_doc(document_name)["sentences"]) - 1
        except AssertionError:
            print(CHECK_FAILED_AT_DOCUMENT, document_name, ", expected",
                  len(internal_parser.get_doc(document_name)["sentences"]), "sentences but", sentence_embedding[-1] + 1, "found")

        # Check if entities correctly embedded
        entity_embedding = data_frame["entity_embedding"].tolist()
        cnt = 0
        for entity_key in entity_position:
            low, high = entity_position[entity_key]
            cnt += high - low
            if high == low:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'entity_embedding', key", entity_key,
                      "is empty (from", low, "to", high, ")")
            else:
                try:
                    assert min(entity_embedding[low:high]) == max(entity_embedding[low:high])
                except AssertionError:
                    print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'entity_embedding', key", entity_key,
                          ", values from", low, "to", high, ":", entity_embedding[low:high], "are inconsistent")
        try:
            assert cnt == (np.array(entity_embedding) != 0).astype(int).sum()
        except AssertionError:
            print(CHECK_FAILED_AT_DOCUMENT, document_name, "in total entity embedded tokens",
                  (np.array(entity_embedding) != 0).astype(int).sum(), "does not match the record", cnt)

        # Check if all relations are valid
        for value in relations.values():
            first = value["source"]
            second = value["target"]
            try:
                assert first in entity_position
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'relations',", first,
                      "is not found in record")
            try:
                assert second in entity_position
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'relations',", second,
                      "is not found in record")


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
        unknown_token_count += data_frame[data_frame["token_ids"] == internal_parser.UNK_TOKEN]["token_ids"].count()
        unknown_entity_token_count += data_frame[(data_frame["entity_embedding"] > 0) &
                                                 (data_frame["token_ids"] == internal_parser.UNK_TOKEN)]["token_ids"].count()
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
        relations = item["relations"]
        for value in relations.values():
            first = value["source"]
            second = value["target"]
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
    tikzplotlib.save("relation.tex")
    plt.show()


def describe_data(docs):
    """Show information about the dataset
    Refer to describe_text_length(), describe_type(), and describe_token()
    """
    describe_text_length(docs)
    print()

    entity_encode = describe_type("mentions", docs)
    print("Entity encoding:", entity_encode)
    print()

    relation_encode = describe_type("relations", docs)
    print("Relation encoding:", relation_encode)
    print()

    data = internal_parser.extract_data_util(docs)
    describe_token(data)
    print()
    describe_relation(data)
    print()


@pytest.mark.xfail(not os.environ.get("GITLAB", 0) == 1)
def test_read_all():
    # # Test main functions on the whole dataset (will take a minute)
    print("-----------------------------------------------------------------------------------------------------------")
    print("Checking the dataset...")
    check_data(internal_parser.get_docs("All"))
    print("-----------------------------------------------------------------------------------------------------------")
    print("Testing data parser and checking parsed all data...")
    data = internal_parser.extract_data("All")
    check_extracted_data(data)
    print("-----------------------------------------------------------------------------------------------------------")
    print("Testing data parser and checking parsed training data...")
    training_data = internal_parser.extract_data("Training")
    check_extracted_data(training_data)
    print("-----------------------------------------------------------------------------------------------------------")
    print("Testing data parser and checking parsed test data...")
    test_data = internal_parser.extract_data("Test")
    check_extracted_data(test_data)
    print("-----------------------------------------------------------------------------------------------------------")
    print("Describing the dataset...")
    describe_data(internal_parser.get_docs("All"))
    print("-----------------------------------------------------------------------------------------------------------")


# -- TEST ------------------------------------------------------------------------------------------------------------ #
# Verify token
def test_UNK_token():
    print("Verify [UNK] token")
    assert internal_parser.get_token_id(["[UNK]"]) == [[internal_parser.UNK_TOKEN]]


def test_CLS_token():
    print("Verify [CLS] token")
    assert internal_parser.get_token_id(["[CLS]"]) == [[internal_parser.CLS_TOKEN]]


def test_SEP_token():
    print("Verify [SEP] token")
    assert internal_parser.get_token_id(["[SEP]"]) == [[internal_parser.SEP_TOKEN]]


# -- TEST ------------------------------------------------------------------------------------------------------------ #
# tmp = internal_parser.DATA_PATH
# internal_parser.DATA_PATH = os.path.dirname(__file__) + ("/" if len(os.path.dirname(__file__)) > 0 else "")
test_doc = "internal_test_doc"


def test_parsing_test_document():
    print("Test parsing the test document...")
    my_new_path = os.path.dirname(__file__) + ("/" if len(os.path.dirname(__file__)) > 0 else "")
    with PathOverWrite(path_data=my_new_path):
        test_words, test_begins, test_ends, test_sentence_embedding = internal_parser.get_word_doc(test_doc)
        test_token_ids = internal_parser.get_token_id(test_words)
        # print("Words:", test_words)
        # print("Token ids:", test_token_ids)

        test_token_ids, test_words, test_begins, test_ends, test_sentence_embedding = \
            internal_parser.expand_token_id(test_token_ids, test_words, test_begins, test_ends, test_sentence_embedding)
        # print("Expanded words:", test_words)
        # print("Expanded token ids:", test_token_ids)

        test_entity_position, test_entity_embedding = internal_parser.get_entity_doc(test_doc, test_begins)
        # print("Entity embedding:", test_entity_embedding)

        # print("Entity tokens:")
        # for low, high in test_entity_position.values():
        #     print(test_words[low:high], test_token_ids[low:high], test_entity_embedding[low:high])

        test_relations = internal_parser.get_relation_doc(test_doc)
        # print("Relations:", test_relations)

        # -- TEST ---------------------------------------------------------------------------------------------------- #
        assert len(test_words) \
            == len(test_token_ids) \
            == len(test_begins) \
            == len(test_ends) \
            == len(test_sentence_embedding) \
            == len(test_entity_embedding)

        # Test if begins is increasing
        for i in range(1, len(test_begins)):
            assert test_begins[i] >= test_begins[i - 1]

        # Test if ends is increasing
        for i in range(1, len(test_ends)):
            assert test_ends[i] >= test_ends[i - 1]

        # Test if ends are always greater than begins
        for i in range(len(test_begins)):
            assert test_begins[i] < test_ends[i]

        # Test if sentence embedding are correct
        for i in range(1, len(test_sentence_embedding)):
            assert 0 <= test_sentence_embedding[i] - test_sentence_embedding[i - 1] <= 1
        assert test_sentence_embedding[-1] == len(internal_parser.get_doc(test_doc)["sentences"]) - 1

        # Test if entities correctly embedded
        cnt = 0
        for low, high in test_entity_position.values():
            cnt += high - low
            assert min(test_entity_embedding[low:high]) == max(test_entity_embedding[low:high])
        assert cnt == (np.array(test_entity_embedding) != 0).astype(int).sum()

        # Test if all relations are valid
        for value in test_relations.values():
            first = value["source"]
            second = value["target"]
            assert first in test_entity_position
            assert second in test_entity_position
