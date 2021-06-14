"""
Spert command-line tool.

.. autosummary::
    :toctree:

    get_optimizer_params
    take_first_tokens
    evaluate
    train
    predict
    load_model
"""
import random

import renard_joint.spert.evaluator as evaluator
import renard_joint.spert.model as model
from renard_joint.spert import SpertConfig
import pandas as pd
import torch
import transformers
from transformers import AdamW, BertConfig, BertTokenizer
from tqdm import tqdm

EPOCH_ = "epoch:"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config(argv):
    import sys

    if len(sys.argv) <= 1:
        raise ValueError("Dataset argument not found")

    spert_config = SpertConfig(argv[1])
    constants, input_generator = spert_config.constants, spert_config.input_generator
    return constants, input_generator


def get_optimizer_params(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_params = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': constants.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    return optimizer_params


def take_first_tokens(embedding, words):
    """Take the embedding of the first token of each word"""
    reduced_embedding = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i - 1]:
            reduced_embedding.append(embedding[i])
    return reduced_embedding


def evaluate(entity_label_map,
             entity_classes,
             relation_label_map,
             relation_classes,
             constants,
             input_generator,
             spert_model,
             group):
    spert_model.eval()
    eval_entity_span_pred = []
    eval_entity_span_true = []
    eval_entity_embedding_pred = []
    eval_entity_embedding_true = []
    eval_relation_span_pred = []
    eval_relation_span_true = []
    eval_generator = input_generator.data_generator(group,
                                                    device,
                                                    is_training=False,
                                                    neg_entity_count=constants.neg_entity_count,
                                                    neg_relation_count=constants.neg_relation_count,
                                                    max_span_size=constants.max_span_size)
    eval_dataset = list(eval_generator)
    eval_size = len(eval_dataset)
    for inputs, infos in tqdm(eval_dataset, total=eval_size, desc="Evaluation " + group):
        # forward
        outputs = spert_model(**inputs, is_training=False)
        # retrieve results for evaluation
        eval_entity_span_pred.append(outputs["entity"]["span"])
        eval_entity_span_true.append(infos["entity_span"])
        if not constants.is_overlapping:
            eval_entity_embedding_pred += take_first_tokens(outputs["entity"]["embedding"].tolist(), infos["words"])
            eval_entity_embedding_true += take_first_tokens(infos["entity_embedding"].tolist(), infos["words"])
            assert len(eval_entity_embedding_pred) == len(eval_entity_embedding_true)
        eval_relation_span_pred.append([] if outputs["relation"] is None else outputs["relation"]["span"])
        eval_relation_span_true.append(infos["relation_span"])
    # evaluate & save
    results = pd.concat([
        evaluator.evaluate_span(eval_entity_span_true, eval_entity_span_pred, entity_label_map, entity_classes),
        evaluator.evaluate_results(eval_entity_embedding_true, eval_entity_embedding_pred, entity_label_map,
                                   entity_classes),
        evaluator.evaluate_loose_relation_span(eval_relation_span_true, eval_relation_span_pred, relation_label_map,
                                               relation_classes),
        evaluator.evaluate_span(eval_relation_span_true, eval_relation_span_pred, relation_label_map, relation_classes),
    ], keys=["Entity span", "Entity embedding", "Loose relation", "Strict relation"])
    results.to_csv(constants.model_save_path + "evaluate_" + group + ".csv")
    print(results)


def train(entity_label_map,
          entity_classes,
          relation_label_map,
          relation_classes,
          relation_possibility,
          constants,
          input_generator):
    """Train the model and evaluate on the dev dataset
    """
    # Training
    train_generator = input_generator.data_generator(constants.train_dataset, device,
                                                     is_training=True,
                                                     neg_entity_count=constants.neg_entity_count,
                                                     neg_relation_count=constants.neg_relation_count,
                                                     max_span_size=constants.max_span_size)
    train_dataset = list(train_generator)
    random.shuffle(train_dataset)
    train_size = len(train_dataset)
    config = BertConfig.from_pretrained(constants.model_path)
    spert_model = model.SpERT.from_pretrained(constants.model_path,
                                              config=config,
                                              # SpERT model parameters
                                              relation_types=constants.relation_types,
                                              entity_types=constants.entity_types,
                                              width_embedding_size=constants.width_embedding_size,
                                              prop_drop=constants.prop_drop,
                                              freeze_transformer=False,
                                              max_pairs=constants.max_pairs,
                                              is_overlapping=constants.is_overlapping,
                                              relation_filter_threshold=constants.relation_filter_threshold,
                                              relation_possibility=relation_possibility)
    spert_model.to(device)
    optimizer_params = get_optimizer_params(spert_model)
    optimizer = AdamW(optimizer_params, lr=constants.lr, weight_decay=constants.weight_decay, correct_bias=False)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=constants.lr_warmup * train_size * constants.epochs,
                                                             num_training_steps=train_size * constants.epochs)
    for epoch in range(constants.epochs):
        losses = []
        entity_losses = []
        relation_losses = []
        train_entity_pred = []
        train_entity_true = []
        train_relation_pred = []
        train_relation_true = []
        spert_model.zero_grad()
        for inputs, infos in tqdm(train_dataset, total=train_size, desc='Train epoch %s' % epoch):
            spert_model.train()
            # forward
            outputs = spert_model(**inputs, is_training=True)
            # backward
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(spert_model.parameters(), constants.max_grad_norm)
            optimizer.step()
            scheduler.step()
            spert_model.zero_grad()
            # retrieve results for evaluation
            losses.append(loss.item())
            entity_losses.append(outputs["entity"]["loss"])
            if outputs["relation"] is not None:
                relation_losses.append(outputs["relation"]["loss"])

            train_entity_pred += outputs["entity"]["pred"].tolist()
            train_entity_true += inputs["entity_label"].tolist()
            train_relation_pred += [] if outputs["relation"] is None else outputs["relation"]["pred"].tolist()
            train_relation_true += inputs["relation_label"].tolist()
            assert len(train_entity_pred) == len(train_entity_true)
            assert len(train_relation_pred) == len(train_relation_true)

        # evaluate & save checkpoint
        print(EPOCH_, epoch, "average loss:", sum(losses) / len(losses))
        print(EPOCH_, epoch, "average entity loss:", sum(entity_losses) / len(entity_losses))
        print(EPOCH_, epoch, "average relation loss:", sum(relation_losses) / len(relation_losses))
        results = pd.concat([
            evaluator.evaluate_results(train_entity_true, train_entity_pred, entity_label_map, entity_classes),
            evaluator.evaluate_results(train_relation_true, train_relation_pred, relation_label_map, relation_classes)
        ], keys=["Entity", "Relation"])
        results.to_csv(constants.model_save_path + "epoch_" + str(epoch) + ".csv")
    #         evaluate(spert_model, constants.dev_dataset)

    torch.save(spert_model.state_dict(), constants.model_save_path + "epoch_" + str(constants.epochs - 1) + ".model")


def predict(entity_label_map,
            relation_label_map,
            tokenizer,
            constants,
            input_generator,
            spert_model,
            sentences):
    for sentence in sentences:
        word_list = sentence.split()
        words = []
        token_ids = []
        # transform a sentence to a document for prediction
        for word in word_list:
            token_id = tokenizer(word)["input_ids"][1:-1]
            for tid in token_id:
                words.append(word)
                token_ids.append(tid)
        data_frame = pd.DataFrame()
        data_frame["words"] = words
        data_frame["token_ids"] = token_ids
        data_frame["entity_embedding"] = 0
        data_frame["sentence_embedding"] = 0  # for internal datasets
        doc = {"data_frame": data_frame,
               "entity_position": {},  # Suppose to appear in non-overlapping dataset
               "entities": {},  # Suppose to appear in overlapping dataset
               "relations": {}}
        # predict
        inputs, infos = input_generator.doc_to_input(doc, device,
                                                     is_training=False,
                                                     max_span_size=constants.max_span_size)
        outputs = spert_model(**inputs, is_training=False)
        pred_entity_span = outputs["entity"]["span"]
        pred_relation_span = [] if outputs["relation"] is None else outputs["relation"]["span"]
        # print the result
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print("Sentence:", sentence)
        print("Entities: (", len(pred_entity_span), ")")
        for begin, end, entity_type in pred_entity_span:
            print(entity_label_map[entity_type], "|", " ".join(tokens[begin:end]))
        print("Relations: (", len(pred_relation_span), ")")
        for e1, e2, relation_type in pred_relation_span:
            print(relation_label_map[relation_type], "|",
                  " ".join(tokens[e1[0]:e1[1]]), "|",
                  " ".join(tokens[e2[0]:e2[1]]))


def load_model(relation_possibility, constants, checkpoint):
    """Import trained model given the checkpoint number"""
    config = BertConfig.from_pretrained(constants.model_path)
    spert_model = model.SpERT.from_pretrained(constants.model_path,
                                              config=config,
                                              # SpERT model parameters
                                              relation_types=constants.relation_types,
                                              entity_types=constants.entity_types,
                                              width_embedding_size=constants.width_embedding_size,
                                              prop_drop=constants.prop_drop,
                                              freeze_transformer=True,
                                              max_pairs=constants.max_pairs,
                                              is_overlapping=constants.is_overlapping,
                                              relation_filter_threshold=constants.relation_filter_threshold,
                                              relation_possibility=relation_possibility)
    spert_model.to(device)
    state_dict = torch.load(constants.model_save_path + "epoch_" + str(checkpoint) + ".model", map_location=device)
    spert_model.load_state_dict(state_dict, strict=False)
    return spert_model


def data_prep(constants, input_generator, dataset):
    entity_label_map = {v: k for k, v in input_generator.parser.entity_encode.items()}
    entity_classes = list(entity_label_map.keys())
    entity_classes.remove(0)

    relation_label_map = {v: k for k, v in input_generator.parser.relation_encode.items()}
    relation_classes = list(relation_label_map.keys())
    relation_classes.remove(0)

    if dataset == "internal":
        relation_possibility = {
            (3, 5): [0, 0, 0, 1, 0, 0, 0, 0],  # Organization + Location -> IsRelatedTo
            (3, 6): [0, 0, 0, 0, 1, 0, 0, 0],  # Organization + CoalActivity -> HasActivity
            (3, 8): [0, 0, 0, 0, 0, 1, 0, 0],  # Organization + SocialOfficialText -> Recognizes
            (3, 4): [0, 1, 0, 0, 0, 0, 0, 0],  # Organization + CommitmentLevel -> Makes
            (4, 1): [0, 0, 1, 0, 0, 0, 0, 0],  # CommitmentLevel + EnvironmentalIssues -> Of
            (4, 7): [0, 0, 1, 0, 0, 0, 0, 0]  # CommitmentLevel + SocialIssues -> Of
        }
        relation_classes.remove(6)  # remove In
        relation_classes.remove(7)  # remove IsInvolvedIn
    else:
        relation_possibility = None

    tokenizer = BertTokenizer.from_pretrained(constants.model_path)
    input_generator.parser.tokenizer = tokenizer

    return entity_label_map, entity_classes, relation_label_map, relation_classes, tokenizer, relation_possibility


def main(constants, input_generator):
    entity_label_map,\
        entity_classes,\
        relation_label_map, \
        relation_classes,\
        tokenizer,\
        relation_possibility = data_prep(constants, input_generator, sys.argv[1])

    if len(sys.argv) <= 2:
        raise ValueError("No functional argument found")
    elif sys.argv[2] == "train":
        train(entity_label_map,
          entity_classes,
          relation_label_map,
          relation_classes,
          relation_possibility,
          constants,
          input_generator)
    elif sys.argv[2] == "evaluate":
        if len(sys.argv) <= 3:
            raise ValueError("No checkpoint number found")
        else:
            spert_model = load_model(relation_possibility, constants, int(sys.argv[3]))
            evaluate(entity_label_map,
                     entity_classes,
                     relation_label_map,
                     relation_classes,
                     constants,
                     input_generator,
                     spert_model,
                     constants.test_dataset)
    elif sys.argv[2] == "predict":
        if len(sys.argv) <= 3:
            raise ValueError("No checkpoint number found")
        else:
            spert_model = load_model(relation_possibility, constants, int(sys.argv[3]))
            predict(entity_label_map,
                    relation_label_map,
                    tokenizer,
                    constants,
                    input_generator,
                    spert_model,
                    sentences=sys.argv[4:])
    else:
        raise ValueError("Invalid argument(s)")


if __name__ == "__main__":
    import sys
    constants, input_generator = config(sys.argv)
    main(constants, input_generator)
