import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

import renard_joint.parser.internal_parser as parser
from renard_joint.spert import evaluator
from renard_joint.spert import model

label_map_bio = {}
for key in parser.entity_encode:
    if parser.entity_encode[key] == 0:
        label_map_bio[0] = "O"
    else:
        label_map_bio[parser.entity_encode[key]] = "B-" + key
        label_map_bio[-parser.entity_encode[key]] = "I-" + key

entity_label_map = {v: k for k, v in parser.entity_encode.items()}
entity_classes = list(entity_label_map.keys())
entity_classes.remove(0)

relation_label_map = {v: k for k, v in parser.relation_encode.items()}
relation_classes = list(relation_label_map.keys())
relation_classes.remove(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model = BertModel.from_pretrained('bert-base-uncased')

print("Loading entity recognition model...")
ner_model = pickle.load(open(os.environ["MODEL"] + "ner/internal_nn_1024.model", 'rb'))

print("Loading relation extraction model...")
re_model = model.BertForMre(len(relation_classes) + 1)
re_model.load_state_dict(torch.load(os.environ["MODEL"] + "re/internal_100.model", map_location=device))
re_model.eval()  # Set model for evaluation only
re_model.to(device)


def transform_doc(
        document,
        pretrain_model,
        ignore_index=CrossEntropyLoss().ignore_index,
        max_token_count=512,
        cls_token=parser.CLS_TOKEN,
        sep_token=parser.SEP_TOKEN
):
    """Transform a parsed document with a pre-trained model (BERT)
    Only the first token of each word is labeled, the others are masked as 'ignore_index'
    The label of O is 0
    The label of I is the negation of the corresponding label of B
    """
    transformed_tokens = []
    padding_token_count = 2

    tokens = document["data_frame"]["token_ids"].tolist()
    begins = document["data_frame"]["begins"].tolist()
    ends = document["data_frame"]["ends"].tolist()
    labels = document["data_frame"]["entity_embedding"].tolist()
    words = document["data_frame"]["words"].tolist()
    sentence_embedding = document["data_frame"]["sentence_embedding"].tolist()

    for i in range(len(tokens)):
        if i > 0 and begins[i] == begins[i - 1] and ends[i] == ends[i - 1]:
            # Extra tokens from the same word are ignored
            labels[i] = ignore_index

    for entity in document["entity_position"]:
        begin, end = document["entity_position"][entity]
        for i in range(begin + 1, end):
            # Every subsequence word of an entity is label as I instead of B
            if labels[i] != ignore_index:
                labels[i] = -labels[i]

    i = 0
    while i < len(tokens):
        j = i
        while (j < len(tokens) and sentence_embedding[i] == sentence_embedding[j] and
               j - i < max_token_count - padding_token_count):
            j += 1
        # Segment the document and encode with the pre-trained model
        inputs = tokens[i:j]
        # Add CLS and SEP tokens
        inputs = [cls_token] + inputs
        inputs.append(sep_token)
        # RUn pretrained model
        outputs = pretrain_model(
            input_ids=torch.tensor([inputs]),
            token_type_ids=torch.tensor([[0] * len(inputs)]),
            attention_mask=torch.tensor([[1] * len(inputs)])
        )
        transformed_tokens += outputs.last_hidden_state[0, 1:-1].tolist()
        i = j

    assert len(transformed_tokens) == len(labels) == len(words)
    return pd.DataFrame(transformed_tokens), pd.DataFrame(list(zip(labels, words)), columns=["labels", "words"])


def predict_entity(ner_model, tokens, labels,
                   ignore_index=CrossEntropyLoss().ignore_index):
    """Given a document, runs entity recognition, returns the predicted entity embedding and spans"""
    true_entity_embedding = np.zeros(tokens.shape[0])
    pred_entity_embedding = np.zeros(tokens.shape[0])
    true_entity_span = []
    pred_entity_span = []
    true_entity_span_lock = True
    pred_entity_span_lock = True

    test_tokens = tokens[labels["labels"] != ignore_index]
    pred_labels = ner_model.predict(test_tokens)

    j = -1
    true_label = None
    pred_label = None
    for i in range(tokens.shape[0]):
        if labels.at[i, "labels"] != ignore_index:
            j += 1
            true_label = labels.at[i, "labels"]
            pred_label = pred_labels[j]

            true_entity_type = label_map_bio[true_label]
            if true_entity_type.startswith("B") or (true_entity_type.startswith("I") and true_entity_span_lock):
                true_entity_span.append((i, i + 1, abs(true_label)))
                true_entity_span_lock = False
            elif true_entity_type == "O":
                true_entity_span_lock = True

            pred_entity_type = label_map_bio[pred_label]
            if pred_entity_type.startswith("B") or (pred_entity_type.startswith("I") and pred_entity_span_lock):
                pred_entity_span.append((i, i + 1, abs(pred_label)))
                pred_entity_span_lock = False
            elif pred_entity_type == "O":
                pred_entity_span_lock = True

            true_entity_embedding[i] = true_label
            pred_entity_embedding[i] = pred_label

        if not true_entity_span_lock:
            if abs(true_label) != abs(true_entity_span[-1][2]):
                true_entity_span.append((i, i + 1, abs(true_label)))
            else:
                true_entity_span[-1] = (true_entity_span[-1][0], i + 1, abs(true_label))

        if not pred_entity_span_lock:
            if abs(pred_label) != abs(pred_entity_span[-1][2]):
                pred_entity_span.append((i, i + 1, abs(pred_label)))
            else:
                pred_entity_span[-1] = (pred_entity_span[-1][0], i + 1, abs(pred_label))

    return np.abs(true_entity_embedding), np.abs(pred_entity_embedding), true_entity_span, pred_entity_span


def get_true_relation_span(doc):
    true_relation_span = []
    for relation in doc["relations"]:
        source = doc["relations"][relation]["source"]
        target = doc["relations"][relation]["target"]
        relation_type = doc["relations"][relation]["type"]

        e1_begin = doc["entity_position"][source][0]
        e1_end = doc["entity_position"][source][1]
        e1_type = doc["data_frame"].at[e1_begin, "entity_embedding"]

        e2_begin = doc["entity_position"][target][0]
        e2_end = doc["entity_position"][target][1]
        e2_type = doc["data_frame"].at[e2_begin, "entity_embedding"]

        true_relation_span.append(((e1_begin, e1_end, e1_type),
                                   (e2_begin, e2_end, e2_type),
                                   relation_type))
    return true_relation_span


def generate_entity_mask(sentence_length, entity_span, offset):
    e1_mask = []
    e2_mask = []
    candidate_relation_span = []
    for e1 in entity_span:
        for e2 in entity_span:
            if e1 != e2:
                template = [0] * sentence_length
                template[e1[0]: e1[1]] = [1] * (e1[1] - e1[0])
                e1_mask.append(template)

                template = [0] * sentence_length
                template[e2[0]: e2[1]] = [1] * (e2[1] - e2[0])
                e2_mask.append(template)

                candidate_relation_span.append(((e1[0] + offset, e1[1] + offset, e1[2]),
                                                (e2[0] + offset, e2[1] + offset, e2[2])))

    return torch.tensor(e1_mask, dtype=torch.long), torch.tensor(e2_mask, dtype=torch.long), candidate_relation_span


def prepare_doc(doc, pred_entity_span, max_token_count=512, max_entity_pair=1000):
    """Prepare inputs for the relation extraction"""
    sentence_id = 0
    starting_index = 0
    input_ids = []
    # ddd a final row with dummy sentence embedding
    doc["data_frame"].loc[doc["data_frame"].index.max() + 1, "sentence_embedding"] \
        = doc["data_frame"]["sentence_embedding"].max() + 1
    for index, row in doc["data_frame"].iterrows():
        if row["sentence_embedding"] != sentence_id or index - starting_index >= max_token_count - 2:
            offset = starting_index - 1  # -1: space for CLS token
            new_entity_span = []
            for entity in pred_entity_span:
                if starting_index <= entity[0] < entity[1] <= index:
                    new_entity_span.append((entity[0] - offset,
                                            entity[1] - offset,
                                            entity[2]))

            # If this sentence has at least two entities for a possible relation
            if len(new_entity_span) >= 2:
                # Add CLS and SEP to the sentence
                input_ids = [parser.CLS_TOKEN] + input_ids + [parser.SEP_TOKEN]
                e1_mask, e2_mask, candidate_relation_span = generate_entity_mask(len(input_ids), new_entity_span,
                                                                                 offset)
                # print(candidate_relation_span)
                assert e1_mask.shape[0] == e2_mask.shape[0] == len(candidate_relation_span)
                assert len(input_ids) == e1_mask.shape[1] == e2_mask.shape[1]
                for i in range(0, e1_mask.shape[0], max_entity_pair):
                    yield {"input_ids": torch.tensor([input_ids]).long().to(device),
                           "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(device),
                           "token_type_ids": torch.zeros((1, len(input_ids)), dtype=torch.long).to(device),
                           "e1_mask": e1_mask[i: min(i + max_entity_pair, e1_mask.shape[0])].to(device),
                           "e2_mask": e2_mask[i: min(i + max_entity_pair, e1_mask.shape[0])].to(device)},\
                          {"offset": offset,
                           "candidate_relation_span": candidate_relation_span[i:
                                                                              min(i + max_entity_pair,
                                                                                  e1_mask.shape[0])]}

            sentence_id = row["sentence_embedding"]
            input_ids = []
            starting_index = index

        input_ids.append(row["token_ids"])


def predict_relation(re_model, doc, pred_entity_span, max_token_count=512, max_entity_pair=1000):
    """Predict the relation type in a document given the predicted entity spans"""
    pred_relation_span = []
    data_generator = prepare_doc(doc, pred_entity_span, max_token_count, max_entity_pair)
    for inputs, infos in data_generator:
        outputs = re_model(**inputs)
        pred_label = F.softmax(outputs.logits, dim=-1).argmax(dim=1)
        # print(pred_label)
        for i in range(pred_label.shape[0]):
            if pred_label[i] != 0:
                candidate_relation = infos["candidate_relation_span"][i]
                pred_relation_span.append((candidate_relation[0], candidate_relation[1], pred_label[i].item()))
    return pred_relation_span


def evaluate(group, bert_model, ner_model, re_model,
             entity_label_map, entity_classes,
             relation_label_map, relation_classes,
             max_token_count=512, max_entity_pair=1000):
    true_entity_embeddings = []
    pred_entity_embeddings = []
    true_entity_spans = []
    pred_entity_spans = []
    true_relation_spans = []
    pred_relation_spans = []

    docs = parser.get_docs(group)
    data = parser.extract_data(docs)
    for doc in tqdm(data, total=len(data), desc="Evaluation " + group):
        token_df, label_df = transform_doc(doc, bert_model, max_token_count=max_token_count)

        # entity recognition
        true_entity_embedding, pred_entity_embedding, true_entity_span, pred_entity_span \
            = predict_entity(ner_model, token_df, label_df)
        true_entity_embeddings += true_entity_embedding.tolist()
        pred_entity_embeddings += pred_entity_embedding.tolist()
        true_entity_spans.append(true_entity_span)
        pred_entity_spans.append(pred_entity_span)

        true_relation_span = get_true_relation_span(doc)
        true_relation_spans.append(true_relation_span)

        # relation extraction
        pred_relation_span = predict_relation(re_model, doc, pred_entity_span,
                                              max_token_count=max_token_count,
                                              max_entity_pair=max_entity_pair)
        pred_relation_spans.append(pred_relation_span)

    results = pd.concat([
        evaluator.evaluate_span(true_entity_spans, pred_entity_spans, entity_label_map, entity_classes),
        evaluator.evaluate_results(true_entity_embeddings, pred_entity_embeddings, entity_label_map, entity_classes),
        evaluator.evaluate_loose_relation_span(true_relation_spans, pred_relation_spans, relation_label_map,
                                               relation_classes),
        evaluator.evaluate_span(true_relation_spans, pred_relation_spans, relation_label_map, relation_classes),
    ], keys=["Entity span", "Entity embedding", "Loose relation", "Strict relation"])
    results.to_csv(os.environ["MODEL"] + "re/internal_evaluate_" + group + ".csv")
    print(results)


def predict(sentences, bert_model, ner_model, re_model,
            entity_label_map,
            relation_label_map,
            max_token_count=512, max_entity_pair=1000):
    for sentence in sentences:
        word_list = sentence.split()
        words = []
        token_ids = []
        begins = []
        ends = []
        # transform a sentence to a document for prediction
        for i, word in enumerate(word_list):
            token_id = parser.tokenizer(word)["input_ids"][1:-1]
            for tid in token_id:
                words.append(word)
                token_ids.append(tid)
                begins.append(i)
                ends.append(i + 1)
        data_frame = pd.DataFrame()
        data_frame["words"] = words
        data_frame["token_ids"] = token_ids
        data_frame["begins"] = begins
        data_frame["ends"] = ends
        data_frame["entity_embedding"] = 0
        data_frame["sentence_embedding"] = 0
        doc = {"data_frame": data_frame,
               "entity_position": {},  # Suppose to appear in non-overlapping dataset
               "relations": {}}
        # predict
        token_df, label_df = transform_doc(doc, bert_model, max_token_count=max_token_count)

        # entity recognition
        true_entity_embedding, pred_entity_embedding, true_entity_span, pred_entity_span \
            = predict_entity(ner_model, token_df, label_df)

        # relation extraction
        pred_relation_span = predict_relation(re_model, doc, pred_entity_span,
                                              max_token_count=max_token_count,
                                              max_entity_pair=max_entity_pair)
        # print the result
        tokens = parser.tokenizer.convert_ids_to_tokens(token_ids)
        print("Sentence:", sentence)
        print("Entities: (", len(pred_entity_span), ")")
        for begin, end, entity_type in pred_entity_span:
            print(entity_label_map[entity_type], "|", " ".join(tokens[begin:end]))
        print("Relations: (", len(pred_relation_span), ")")
        for e1, e2, relation_type in pred_relation_span:
            print(relation_label_map[relation_type], "|",
                  " ".join(tokens[e1[0]:e1[1]]), "|",
                  " ".join(tokens[e2[0]:e2[1]]))


if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        raise ValueError("No argument found")
    elif sys.argv[1] == "evaluate":
        evaluate("Test", bert_model, ner_model, re_model,
                 entity_label_map, entity_classes, relation_label_map, relation_classes)
    elif sys.argv[1] == "predict":
        predict(sys.argv[2:], bert_model, ner_model, re_model,
                entity_label_map, entity_classes, relation_label_map, relation_classes)
    else:
        raise ValueError("Invalid argument(s)")
