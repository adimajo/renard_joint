import numpy as np
import internal_parser

print("Checking the dataset...")
internal_parser.check_data()

# -- TEST ------------------------------------------------------------------------------------------------------------ #
# Verify [UNK] token
print("Verify [UNK] token")
assert internal_parser.get_token_id(["[UNK]"]) == [[internal_parser.UNK_TOKEN]]

# -- TEST ------------------------------------------------------------------------------------------------------------ #
test_doc = internal_parser.get_record()[0]["id"]
print("The first document is:", test_doc)
assert test_doc == "143f9e00-34c4-11eb-a28a-8b07c9b15060-0"

print("Test parsing this document...")
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

test_relation_position = internal_parser.get_relation_doc(test_doc, test_entity_position)
# print("Relation position:", test_relation_position)

# -- TEST ------------------------------------------------------------------------------------------------------------ #
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
for first, second in test_relation_position.values():
    assert first in test_entity_position
    assert second in test_entity_position

# The next part test main functions on the whole dataset (will take a minute)

# print("Testing data parser and checking parsed data...")
# data = internal_parser.extract_data()
# internal_parser.check_extracted_data(data)

# print("Describing the dataset...")
# internal_parser.describe_data()
