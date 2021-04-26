import scripts.spert as spert
import renard_joint.spert.internal_constants as constants
import renard_joint.spert.internal_input_generator as input_generator

entity_label_map, \
    entity_classes, \
    relation_label_map, \
    relation_classes, \
    tokenizer, \
    relation_possibility = spert.data_prep(constants, input_generator, "internal")

spert_model = spert.load_model(relation_possibility, constants, 26)


def test_evaluate_conll_spert():
    spert.evaluate(entity_label_map,
                   entity_classes,
                   relation_label_map,
                   relation_classes,
                   constants,
                   input_generator,
                   spert_model,
                   constants.test_dataset)


def test_predict_conll_spert():
    spert.predict(entity_label_map,
                  relation_label_map,
                  tokenizer,
                  constants,
                  input_generator,
                  spert_model,
                  ["Adrien is testing the Data Harvesting prototype"])