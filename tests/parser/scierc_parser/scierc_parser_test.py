import matplotlib.pyplot as plt
import seaborn as sns

from renard_joint.parser import scierc_parser

CHECK_FAILED_AT_DOCUMENT = "Check failed at document"

DOCUMENT_ = "Document '"


def check_doc(document):
    """Check if the data in the document is consistent
    .. todo: move this to tests
    """
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
        print(DOCUMENT_, doc_id, "' does not have key 'sentences'")
        return
    try:
        assert "ner" in document
    except AssertionError:
        print(DOCUMENT_, doc_id, "' does not have key 'ner'")
        return
    try:
        assert "relations" in document
    except AssertionError:
        print(DOCUMENT_, doc_id, "' does not have key 'relations'")
        return
    try:
        assert len(document["sentences"]) == len(document["ner"]) == len(document["relations"])
    except AssertionError:
        print(DOCUMENT_, doc_id, "' does not have consistent number of sentences")
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
                print(DOCUMENT_, doc_id, "', entity", entity, "is inconsistent")
    # Check relation consistency
    for i in range(len(document["relations"])):
        for relation in document["relations"][i]:
            try:
                assert len(relation) == 5
                assert isinstance(relation[4], str)
                assert 0 <= relation[0] <= relation[1] < total_length
                assert 0 <= relation[2] <= relation[3] < total_length
            except AssertionError:
                print(DOCUMENT_, doc_id, "', relation", relation, "is inconsistent")


def check_docs(group):
    """Check if all the documents contained in the data group is consistent
    'group' is either "train", "dev", or "test"
    """
    docs = scierc_parser.get_docs(group)
    for document in docs:
        check_doc(document)


def check_data():
    """Check if the everything in the dataset is consistent
    Refer to check_docs(group) and check_doc(document)
    .. todo: move this to tests
    """
    check_docs("train")
    check_docs("dev")
    check_docs("test")


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
    docs = scierc_parser.get_docs(group)
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
    docs = scierc_parser.get_docs(group)
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
    docs = scierc_parser.get_docs(group)
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


def check_extracted_data(data):
    """Check if all extracted data is valid
    .. todo: move this to tests
    """
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
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'sentence_embedding' at position",
                      i, "sentence_embedding[i] - sentence_embedding[i-1] =",
                      sentence_embedding[i] - sentence_embedding[i - 1])

        # Check if all relations are valid
        for value in relations.values():
            first = value["source"]
            second = value["target"]
            try:
                assert first in entities
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'relations',", first,
                      "is not found in record")
            try:
                assert second in entities
            except AssertionError:
                print(CHECK_FAILED_AT_DOCUMENT, document_name, "in 'relations',", second,
                      "is not found in record")


# # Test main functions on the whole dataset (will take a minute)
print("-------------------------------------------------------------------------------------------------------------")
print("Checking the dataset...")
check_data()
print("-------------------------------------------------------------------------------------------------------------")
print("Testing data parser and checking parsed training data...")
training_data = scierc_parser.extract_data(scierc_parser.get_docs("train"))
check_extracted_data(training_data)
print("-------------------------------------------------------------------------------------------------------------")
print("Testing data parser and checking parsed dev data...")
data = scierc_parser.extract_data(scierc_parser.get_docs("dev"))
check_extracted_data(data)
print("-------------------------------------------------------------------------------------------------------------")
print("Testing data parser and checking parsed test data...")
test_data = scierc_parser.extract_data(scierc_parser.get_docs("test"))
check_extracted_data(test_data)
print("-------------------------------------------------------------------------------------------------------------")
print("Describing the dataset...")
describe_data()
print("-------------------------------------------------------------------------------------------------------------")


# -- TEST ------------------------------------------------------------------------------------------------------------ #
# Verify token
def test_UNK_token():
    print("Verify [UNK] token")
    assert scierc_parser.get_token_id(["[UNK]"]) == [[scierc_parser.UNK_TOKEN]]


def test_CLS_token():
    print("Verify [CLS] token")
    assert scierc_parser.get_token_id(["[CLS]"]) == [[scierc_parser.CLS_TOKEN]]


def test_SEP_token():
    print("Verify [SEP] token")
    assert scierc_parser.get_token_id(["[SEP]"]) == [[scierc_parser.SEP_TOKEN]]


def test_parsing_test_document():
    print("Test parsing the test documents...")
    list_of_docs_to_data = scierc_parser.extract_data("test")
    list_of_docs = scierc_parser.get_docs("test")
    data_dict = scierc_parser.extract_doc(list_of_docs[0])
    doc_id = data_dict["document"]
    data_frame = data_dict["data_frame"]
    entity_span = data_dict["entities"]
    relations = data_dict["relations"]

    print("Doc id:", doc_id)
    print("Words:", data_frame['words'])
    print("Token ids:", data_frame['token_ids'])


# def test_exception():
#     scierc_parser.expand_token_id(token_ids, words, sentence_embedding, entities)
