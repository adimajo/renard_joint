import pytest

import renard_joint.spert.conll04_constants as constants
import renard_joint.spert.conll04_input_generator as input_generator
import renard_joint._scripts.spert as spert


try:
    entity_label_map, \
        entity_classes, \
        relation_label_map, \
        relation_classes, \
        tokenizer, \
        relation_possibility = spert.data_prep(constants, input_generator, "conll04")

    spert_model = spert.load_model(relation_possibility, constants, 19)
except:
    pass


@pytest.mark.xfail
def test_evaluate_conll_spert():
    spert.evaluate(entity_label_map,
                   entity_classes,
                   relation_label_map,
                   relation_classes,
                   constants,
                   input_generator,
                   spert_model,
                   constants.test_dataset)


@pytest.mark.xfail
def test_predict_conll_spert():
    spert.predict(entity_label_map,
                  relation_label_map,
                  tokenizer,
                  constants,
                  input_generator,
                  spert_model,
                  ["Adrien is testing the Data Harvesting prototype"])
