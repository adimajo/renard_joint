import os

import pytest

import renard_joint._scripts.spert as spert
from renard_joint.spert import conll04_constants
from renard_joint.spert import conll04_input_generator

try:
    entity_label_map, \
        entity_classes, \
        relation_label_map, \
        relation_classes, \
        tokenizer, \
        relation_possibility = spert.data_prep(conll04_constants,
                                               conll04_input_generator,
                                               "conll04")

    spert_model = spert.load_model(relation_possibility,
                                   conll04_constants,
                                   19)
except:
    pass


@pytest.mark.xfail(not os.environ.get("GITLAB", 0) == 1, reason="Not on Gitlab")
def test_evaluate_conll_spert():
    spert.evaluate(entity_label_map,
                   entity_classes,
                   relation_label_map,
                   relation_classes,
                   conll04_constants,
                   conll04_input_generator,
                   spert_model,
                   conll04_constants.test_dataset)


@pytest.mark.xfail(not os.environ.get("GITLAB", 0) == 1, reason="Not on Gitlab")
def test_predict_conll_spert():
    spert.predict(entity_label_map,
                  relation_label_map,
                  tokenizer,
                  conll04_constants,
                  conll04_input_generator,
                  spert_model,
                  ["Adrien is testing the Data Harvesting prototype"])
