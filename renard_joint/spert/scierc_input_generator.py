import os
import math
import random
import torch

import sys
sys.path.append("../parser")
import scierc_parser as parser


def generate_entity_mask(doc, is_training, neg_entity_count, max_span_size):
    sentence_length = doc["data_frame"].shape[0]
    entity_pool = set([(l, r) for l in range(sentence_length) \
                       for r in range(l + 1, min(sentence_length, l + max_span_size) + 1)])
    # print(sorted(entity_pool))
    entity_mask = []
    entity_label = []
    entity_span = []
    
    for key in doc["entities"]:
        l = doc["entities"][key]["begin"]
        r = doc["entities"][key]["end"]
        t = doc["entities"][key]["type"]
        if r - l <= max_span_size: entity_pool.remove((l, r))
        entity_mask.append([0] * l + [1] * (r - l) + [0] * (sentence_length - r))
        entity_label.append(t)
        entity_span.append((l, r, t))
        
    if is_training:
        # If training then add a limited number of negative spans
        for l, r in random.sample(entity_pool, min(len(entity_pool), neg_entity_count)):
            entity_mask.append([0] * l + [1] * (r - l) + [0] * (sentence_length - r))
            entity_label.append(0)
    else:
        # Else add all possible negative spans
        for l, r in entity_pool:
            entity_mask.append([0] * l + [1] * (r - l) + [0] * (sentence_length - r))
            entity_label.append(0)
    
    if len(entity_mask) > 1:
        # if there is more than 1 entity, random shuffle
        tmp = list(zip(entity_mask, entity_label))
        random.shuffle(tmp)
        entity_mask, entity_label = zip(*tmp)
            
    return torch.tensor(entity_mask, dtype=torch.long), torch.tensor(entity_label, dtype=torch.long), entity_span


def generate_relation_mask(doc, is_training, neg_relation_count):
    sentence_length = doc["data_frame"].shape[0]
    relation_pool = set([(e1, e2) for e1 in doc["entities"].keys() \
                       for e2 in doc["entities"].keys() if e1 != e2])
    # print(relation_pool)
    relation_mask = []
    relation_label = []
    relation_span = []
    
    # print(doc["relations"])
    for key in doc["relations"]:
        if (doc["relations"][key]["source"], doc["relations"][key]["target"]) in relation_pool:
            relation_pool.discard((doc["relations"][key]["source"], doc["relations"][key]["target"]))
            e1 = doc["entities"][doc["relations"][key]["source"]]
            e2 = doc["entities"][doc["relations"][key]["target"]]
            c = (min(e1["end"], e2["end"]), max(e1["begin"], e2["begin"]))
            template = [1] * sentence_length
            template[e1["begin"]: e1["end"]] = [x*2 for x in template[e1["begin"]: e1["end"]]]
            template[e2["begin"]: e2["end"]] = [x*3 for x in template[e2["begin"]: e2["end"]]]
            template[c[0]: c[1]] = [x*5 for x in template[c[0]: c[1]]]
            relation_mask.append(template)        
            relation_label.append(doc["relations"][key]["type"])
            relation_span.append(((e1["begin"], e1["end"], e1["type"]), 
                                  (e2["begin"], e2["end"], e2["type"]), 
                                  doc["relations"][key]["type"]))
            
    for key in doc["relations"]:
        relation_pool.discard((doc["relations"][key]["target"], doc["relations"][key]["source"])) # remove reverse
        
    # Only use real entities to generate false relations (refer to the paper)
    if is_training:
        # Only add negative relations when training
        for first, second in random.sample(relation_pool, min(len(relation_pool), neg_relation_count)):
            e1 = doc["entities"][first]
            e2 = doc["entities"][second]
            c = (min(e1["end"], e2["end"]), max(e1["begin"], e2["begin"]))
            template = [1] * sentence_length
            template[e1["begin"]: e1["end"]] = [x*2 for x in template[e1["begin"]: e1["end"]]]
            template[e2["begin"]: e2["end"]] = [x*3 for x in template[e2["begin"]: e2["end"]]]
            template[c[0]: c[1]] = [x*5 for x in template[c[0]: c[1]]]
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

    return {
        "input_ids": torch.tensor([input_ids]).long().to(device), 
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(device),
        "token_type_ids": torch.zeros((1, len(input_ids)), dtype=torch.long).to(device),
        "entity_mask": entity_mask.to(device),
        "entity_label": entity_label.to(device),
        "relation_mask": relation_mask.to(device),
        "relation_label": relation_label.to(device)
    }, {
        # Add information to trace back and evaluate
        "words": doc["data_frame"]["words"],
        "entity_span": entity_span, # ground truth entity spans
        "relation_span": relation_span # ground truth relation spans
    }


def data_generator(group, device, 
                   is_training=True,
                   neg_entity_count=100, 
                   neg_relation_count=100, 
                   max_span_size=10):
    """Generate input for the spert model
    'group' is the dataset ("train", "dev", or "test")
    'device' is the device where pytorch runs on (e.g. device = torch.device("cuda"))
    """
    data = parser.extract_data(group)
    for doc in data:
        yield doc_to_input(doc, device, is_training, 
                           neg_entity_count, neg_relation_count, max_span_size)
