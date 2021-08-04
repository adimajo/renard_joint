import copy

import pytest

import renard_joint._scripts.spert as spert
from renard_joint.spert import internal_constants
from renard_joint.spert import internal_input_generator

entity_label_map_toto, \
    entity_classes_toto, \
    relation_label_map_toto, \
    relation_classes_toto, \
    tokenizer_toto, \
    relation_possibility_toto = spert.data_prep(internal_constants,
                                                internal_input_generator,
                                                "internal")

spert_model_toto = spert.load_model(relation_possibility_toto,
                                    internal_constants,
                                    26)


# @pytest.mark.xfail(not os.environ.get("GITLAB", 0) == 1, reason="Not on Gitlab")
def test_evaluate_internal_spert():
    # spert_model_toto.resize_token_embeddings(len(tokenizer_toto))
    spert.evaluate(entity_label_map_toto,
                   entity_classes_toto,
                   relation_label_map_toto,
                   relation_classes_toto,
                   internal_constants,
                   internal_input_generator,
                   spert_model_toto,
                   copy.deepcopy(internal_constants.test_dataset))


def test_predict_internal_spert():
    spert.predict(entity_label_map_toto,
                  relation_label_map_toto,
                  tokenizer_toto,
                  internal_constants,
                  internal_input_generator,
                  spert_model_toto,
                  ["Adrien is testing the Data Harvesting prototype"])


def test_spert_config():
    with pytest.raises(ValueError):
        spert.SpertConfig(dataset=None)
    with pytest.raises(ValueError):
        spert.SpertConfig(dataset="toto")
    spert.SpertConfig(dataset="internal")
    spert.SpertConfig(dataset="scierc")
    spert.SpertConfig(dataset="conll04")
