import os
import math
import random
import torch

import sys
sys.path.append("../parser")
import conll04_parser as parser

# B and I represent the same type of entity
parser.entity_encode = {'O': 0, 'B-Loc': 1, 'I-Loc': 1, 'B-Peop': 2, 'I-Peop': 2, 
                 'B-Org': 3, 'I-Org': 3, 'B-Other': 4, 'I-Other': 4}


def generate_entity_mask(doc, is_training, neg_entity_count, max_span_size):
    sentence_length = doc["data_frame"].shape[0]
    entity_pool = set([(l, r) for l in range(sentence_length) \
                       for r in range(l + 1, min(sentence_length, l + max_span_size) + 1)])
    # print(sorted(entity_pool))
    entity_mask = []
    entity_label = []
    
    for key in doc["entity_position"]:
        l, r = doc["entity_position"][key]
        if r - l <= max_span_size: entity_pool.remove((l, r))
        entity_mask.append([0] * l + [1] * (r - l) + [0] * (sentence_length - r))
        entity_label.append(doc["data_frame"].at[l, "entity_embedding"])
        
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
            
    return torch.tensor(entity_mask, dtype=torch.long), torch.tensor(entity_label, dtype=torch.long)


def generate_relation_mask(doc, is_training, neg_relation_count):
    sentence_length = doc["data_frame"].shape[0]
    relation_pool = set([(e1, e2) for e1 in doc["entity_position"].keys() \
                       for e2 in doc["entity_position"].keys() if e1 != e2])
    # print(relation_pool)
    relation_mask = []
    relation_label = []
    
    for key in doc["relations"]:
        relation_pool.remove((doc["relations"][key]["source"], doc["relations"][key]["target"]))
        e1 = doc["entity_position"][doc["relations"][key]["source"]]
        e2 = doc["entity_position"][doc["relations"][key]["target"]]
        c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
        if c[1] > c[0]:
            template = [0] * sentence_length
            template[e1[0]: e1[1]] = [1] * (e1[1] - e1[0])
            template[e2[0]: e2[1]] = [2] * (e2[1] - e2[0])
            template[c[0]: c[1]] = [3] * (c[1] - c[0])
            relation_mask.append(template)        
            relation_label.append(doc["relations"][key]["type"])
        
    # Only use real entities to generate false relations (refer to the paper)
    if is_training:
        # Only add negative relations when training
        for first, second in random.sample(relation_pool, min(len(relation_pool), neg_relation_count)):
            e1 = doc["entity_position"][first]
            e2 = doc["entity_position"][second]
            c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
            if c[1] > c[0]:
                template = [0] * sentence_length
                template[e1[0]: e1[1]] = [1] * (e1[1] - e1[0])
                template[e2[0]: e2[1]] = [2] * (e2[1] - e2[0])
                template[c[0]: c[1]] = [3] * (c[1] - c[0])
                relation_mask.append(template)        
                relation_label.append(0)
    
    return torch.tensor(relation_mask, dtype=torch.long), torch.tensor(relation_label, dtype=torch.long)


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
        # Add CLS and SEP to the sentence
        input_ids = [parser.CLS_TOKEN] + doc["data_frame"]["token_ids"].tolist() + [parser.SEP_TOKEN]
        
        entity_mask, entity_label = generate_entity_mask(doc, is_training, neg_entity_count, max_span_size)
        assert entity_mask.shape[1] == len(input_ids) - 2
        
        relation_mask, relation_label = generate_relation_mask(doc, is_training, neg_relation_count)
        assert relation_mask.shape[1] == len(input_ids) - 2
        
        yield {
            "input_ids": torch.tensor([input_ids]).long().to(device), 
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(device),
            "token_type_ids": torch.zeros((1, len(input_ids)), dtype=torch.long).to(device),
            "entity_mask": entity_mask.to(device),
            "entity_label": entity_label.to(device),
            "relation_mask": relation_mask.to(device),
            "relation_label": relation_label.to(device)
        }
        del input_ids
        del entity_mask
        del entity_label
        del relation_mask
        del relation_label
