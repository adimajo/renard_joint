import scripts.spert as spert


def test_evaluate_internal_pipeline():
    spert.evaluate("test",
                   spert.bert_model,
                   spert.ner_model,
                   spert.re_model,
                   spert.entity_label_map,
                   spert.entity_classes,
                   spert.relation_label_map,
                   spert.relation_classes)


def test_predict_internal_spert():
    spert.predict(["Adrien is testing the Data Harvesting prototype"],
                  spert.bert_model,
                  spert.ner_model,
                  spert.re_model,
                  spert.entity_label_map,
                  spert.relation_label_map)
