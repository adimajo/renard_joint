import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from renard_joint.parser import conll04_parser

CHECK_FAILED_AT_DOCUMENT = "Check failed at document"


def count_entity_doc(document):
    """Count the number of each type of entity in a document"""
    count = {}
    for line in document[1:]:
        _, _, entity_type, _, _ = conll04_parser.split_line(line)
        if entity_type in count:
            count[entity_type] += 1
        else:
            count[entity_type] = 1
    return count


def count_relation_doc(document):
    """Count the number of each type of entity in a document"""
    count = {}
    for line in document[1:]:
        _, _, _, relation_types, _ = conll04_parser.split_line(line)
        for relation in relation_types:
            if relation in count:
                count[relation] += 1
            else:
                count[relation] = 1
    return count


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


def describe_text_length(group):
    """Show information about the length of documents in the dataset group"""
    lengths = []
    docs = conll04_parser.get_docs(group)
    for document in docs:
        lengths.append(conll04_parser.get_text_length_doc(document))
    describe_list(lengths, "document lengths of " + group)


def describe_type(group, type_counter, describe=True):
    """Describe the types of a property in the dataset"""
    docs = conll04_parser.get_docs(group)
    count = {}
    for document in docs:
        cnt = type_counter(document)
        for key in cnt:
            if key not in count:
                count[key] = cnt[key]
            else:
                count[key] += cnt[key]
    if describe:
        print("Description of type in", group)
        print("Total:", sum(count.values()))
        for key in count:
            print(key, ":", count[key])
        sns.barplot(list(count.values()), list(count.keys()))
        plt.show()
    # Return a map from entities to corresponding encoding numbers
    return dict(zip(count.keys(), range(len(count))))


def describe_data():
    print("Description of train dataset:")
    describe_text_length("train")
    describe_type("train", count_entity_doc)
    describe_type("train", count_relation_doc)
    print("---------------------------------------------------------------------------------")
    print("Description of dev dataset:")
    describe_text_length("dev")
    describe_type("dev", count_entity_doc)
    describe_type("dev", count_relation_doc)
    print("---------------------------------------------------------------------------------")
    print("Description of test dataset:")
    describe_text_length("test")
    describe_type("test", count_entity_doc)
    describe_type("test", count_relation_doc)
    print("---------------------------------------------------------------------------------")


def check_doc(document):
    """Check if the data in the document is consistent
    """
    try:
        assert document[0].startswith("#doc ")
    except AssertionError:
        print("The document does not start with '#doc' but instead", document[0])
    doc_id = document[0].split()[1]
    for i in range(1, len(document)):
        line = document[i]
        try:
            assert int(line.split()[0]) == i - 1
        except (ValueError, AssertionError):
            print("Document", doc_id, "line", i, ":", line, "expect line index",
                  i - 1, ", found", line.split()[0])


def check_docs(group):
    """Check if all the documents contained in the data group is consistent
    'group' is either "train", "dev", or "test"
    """
    docs = conll04_parser.get_docs(group)
    for document in docs:
        check_doc(document)


def check_data():
    """Check if the everything in the dataset is consistent
    Refer to check_docs(group) and check_doc(document)
    """
    check_docs("train")
    check_docs("dev")
    check_docs("test")


def check_extracted_data(data):
    """Check if all extracted data is valid
    """
    for item in data:
        document_name = item["document"]
        data_frame = item["data_frame"]
        entity_position = item["entity_position"]
        relations = item["relations"]

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
                    assert abs(min(entity_embedding[low:high]) - max(entity_embedding[low:high])) <= 1
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


# # Test main functions on the whole dataset (will take a minute)
print("-------------------------------------------------------------------------------------------------------------")
print("Checking the dataset...")
check_data()
print("-------------------------------------------------------------------------------------------------------------")
print("Testing data parser and checking parsed all data...")
data = conll04_parser.extract_data(conll04_parser.get_docs("All"))
check_extracted_data(data)
print("-------------------------------------------------------------------------------------------------------------")
print("Testing data parser and checking parsed training data...")
training_data = conll04_parser.extract_data(conll04_parser.get_docs("Training"))
check_extracted_data(training_data)
print("-------------------------------------------------------------------------------------------------------------")
print("Testing data parser and checking parsed test data...")
test_data = conll04_parser.extract_data(conll04_parser.get_docs("Test"))
check_extracted_data(test_data)
print("-------------------------------------------------------------------------------------------------------------")
print("Describing the dataset...")
describe_data()
print("-------------------------------------------------------------------------------------------------------------")


# -- TEST ------------------------------------------------------------------------------------------------------------ #
# Verify token
def test_UNK_token():
    print("Verify [UNK] token")
    assert conll04_parser.get_token_id(["[UNK]"]) == [[conll04_parser.UNK_TOKEN]]


def test_CLS_token():
    print("Verify [CLS] token")
    assert conll04_parser.get_token_id(["[CLS]"]) == [[conll04_parser.CLS_TOKEN]]


def test_SEP_token():
    print("Verify [SEP] token")
    assert conll04_parser.get_token_id(["[SEP]"]) == [[conll04_parser.SEP_TOKEN]]


def test_parsing_test_document():
    print("Test parsing the test documents...")
    list_of_docs_to_data = conll04_parser.extract_data("test")
    list_of_docs = conll04_parser.get_docs("test")
    data_dict = conll04_parser.extract_doc(list_of_docs[0])
    doc_id = data_dict["document"]
    data_frame = data_dict["data_frame"]
    entity_position = data_dict["entity_position"]
    relations = data_dict["relations"]

    print("Doc id:", doc_id)
    print("Words:", data_frame['words'])
    print("Token ids:", data_frame['token_ids'])
