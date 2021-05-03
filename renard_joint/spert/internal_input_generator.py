"""
Input data generator for internal data.

.. autosummary::
    :toctree:

    generate_entity_mask
    generate_relation_mask
    doc_to_input
    data_generator
"""
import random

import torch

import renard_joint.parser.internal_parser as parser


def generate_entity_mask(doc, is_training, neg_entity_count, max_span_size):
    sentence_length = doc["data_frame"].shape[0]
    entity_pool = set()
    for index_word in range(sentence_length):
        if index_word == 0 or doc["data_frame"].at[index_word,
                                                   "words"] != doc["data_frame"].at[index_word - 1, "words"]:
            i = 0
            for r in range(index_word + 1, sentence_length + 1):
                if r == sentence_length or doc["data_frame"].at[r, "words"] != doc["data_frame"].at[r - 1, "words"]:
                    entity_pool.add((index_word, r))
                    i += 1
                    if i >= max_span_size:
                        break  # the span reaches max size limit
    # print(sorted(entity_pool))
    entity_mask = []
    entity_label = []
    entity_span = []

    for key in doc["entity_position"]:
        index_word, r = doc["entity_position"][key]
        entity_pool.discard((index_word, r))
        entity_mask.append([0] * index_word + [1] * (r - index_word) + [0] * (sentence_length - r))
        entity_label.append(doc["data_frame"].at[index_word, "entity_embedding"])
        entity_span.append((index_word, r, doc["data_frame"].at[index_word, "entity_embedding"]))

    if is_training:
        # If training then add a limited number of negative spans
        for index_word, r in random.sample(entity_pool, min(len(entity_pool), neg_entity_count)):
            entity_mask.append([0] * index_word + [1] * (r - index_word) + [0] * (sentence_length - r))
            entity_label.append(0)
    else:
        # Else add all possible negative spans
        for index_word, r in entity_pool:
            entity_mask.append([0] * index_word + [1] * (r - index_word) + [0] * (sentence_length - r))
            entity_label.append(0)

    if len(entity_mask) > 1:
        # if there is more than 1 entity, random shuffle
        tmp = list(zip(entity_mask, entity_label))
        random.shuffle(tmp)
        entity_mask, entity_label = zip(*tmp)

    return torch.tensor(entity_mask, dtype=torch.long), torch.tensor(entity_label, dtype=torch.long), entity_span


def generate_relation_mask(doc, is_training, neg_relation_count):
    sentence_length = doc["data_frame"].shape[0]
    relation_pool = set([(e1, e2) for e1 in doc["entity_position"].keys()
                         for e2 in doc["entity_position"].keys() if e1 != e2])
    # print(relation_pool)
    relation_mask = []
    relation_label = []
    relation_span = []

    for key in doc["relations"]:
        relation_pool.discard((doc["relations"][key]["source"], doc["relations"][key]["target"]))
        relation_pool.discard((doc["relations"][key]["target"], doc["relations"][key]["source"]))  # remove reverse
        e1 = doc["entity_position"][doc["relations"][key]["source"]]
        e2 = doc["entity_position"][doc["relations"][key]["target"]]
        c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
        template = [1] * sentence_length
        template[e1[0]: e1[1]] = [x * 2 for x in template[e1[0]: e1[1]]]
        template[e2[0]: e2[1]] = [x * 3 for x in template[e2[0]: e2[1]]]
        template[c[0]: c[1]] = [x * 5 for x in template[c[0]: c[1]]]
        relation_mask.append(template)
        relation_label.append(doc["relations"][key]["type"])
        relation_span.append(((e1[0], e1[1], doc["data_frame"].at[e1[0], "entity_embedding"]),
                              (e2[0], e2[1], doc["data_frame"].at[e2[0], "entity_embedding"]),
                              doc["relations"][key]["type"]))

    # Only use real entities to generate false relations (refer to the paper)
    if is_training:
        # Only add negative relations when training
        for first, second in random.sample(relation_pool, min(len(relation_pool), neg_relation_count)):
            e1 = doc["entity_position"][first]
            e2 = doc["entity_position"][second]
            c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
            template = [1] * sentence_length
            template[e1[0]: e1[1]] = [x * 2 for x in template[e1[0]: e1[1]]]
            template[e2[0]: e2[1]] = [x * 3 for x in template[e2[0]: e2[1]]]
            template[c[0]: c[1]] = [x * 5 for x in template[c[0]: c[1]]]
            relation_mask.append(template)
            relation_label.append(0)

    if len(relation_mask) > 1:
        # if there is more than 1 relation, random shuffle
        tmp = list(zip(relation_mask, relation_label))
        random.shuffle(tmp)
        relation_mask, relation_label = zip(*tmp)

    return torch.tensor(relation_mask, dtype=torch.long), torch.tensor(relation_label, dtype=torch.long), relation_span


def doc_to_input(doc, device,
                 is_training=True,
                 neg_entity_count=100,
                 neg_relation_count=100,
                 max_span_size=10):
    # Add CLS and SEP to the sentence
    input_ids = [parser.CLS_TOKEN] + doc["data_frame"]["token_ids"].tolist() + [parser.SEP_TOKEN]

    entity_mask, entity_label, entity_span = generate_entity_mask(doc, is_training, neg_entity_count, max_span_size)
    assert entity_mask.shape[1] == len(input_ids) - 2

    relation_mask, relation_label, relation_span = generate_relation_mask(doc, is_training, neg_relation_count)
    if not torch.equal(relation_mask, torch.tensor([], dtype=torch.long)):
        assert relation_mask.shape[1] == len(input_ids) - 2
    return {"input_ids": torch.tensor([input_ids]).long().to(device),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(device),
            "token_type_ids": torch.zeros((1, len(input_ids)), dtype=torch.long).to(device),
            "entity_mask": entity_mask.to(device),
            "entity_label": entity_label.to(device),
            "relation_mask": relation_mask.to(device),
            "relation_label": relation_label.to(device)},\
           {"words": doc["data_frame"]["words"],
            "entity_embedding": doc["data_frame"]["entity_embedding"],
            "entity_span": entity_span,  # ground truth entity spans
            "relation_span": relation_span}  # ground truth relation spans


def data_generator(group,
                   device,
                   is_training=True,
                   neg_entity_count=100,
                   neg_relation_count=100,
                   max_span_size=10):
    """Generate input for the spert model
    'group' is the dataset ("Training" or "Test")
    'device' is the device where pytorch runs on (e.g. device = torch.device("cuda"))
    """
    data = parser.extract_all_data(group)
    for doc in data:
        sentence_id = 0
        starting_index = 0
        # ddd a final row with dummy sentence embedding
        doc["data_frame"].loc[doc["data_frame"].index.max() + 1, "sentence_embedding"] \
            = doc["data_frame"]["sentence_embedding"].max() + 1
        for index, row in doc["data_frame"].iterrows():
            if row["sentence_embedding"] != sentence_id:
                if index - starting_index > 510:
                    starting_index = index - 510
                tmp_entity_position = {}
                for entity in doc["entity_position"]:
                    if starting_index <= doc["entity_position"][entity][0] < doc["entity_position"][entity][1] <= index:
                        tmp_entity_position[entity] = (
                            doc["entity_position"][entity][0] - starting_index,
                            doc["entity_position"][entity][1] - starting_index
                        )
                tmp_relations = {}
                for relations in doc["relations"]:
                    if doc["relations"][relations]["source"] in tmp_entity_position and \
                            doc["relations"][relations]["target"] in tmp_entity_position:
                        tmp_relations[relations] = doc["relations"][relations]
                tmp_doc = {
                    "data_frame": doc["data_frame"][starting_index: index].reset_index(drop=True),
                    "entity_position": tmp_entity_position,
                    "relations": tmp_relations
                }
                yield doc_to_input(tmp_doc, device, is_training,
                                   neg_entity_count, neg_relation_count, max_span_size)
                sentence_id = row["sentence_embedding"]
                starting_index = index
