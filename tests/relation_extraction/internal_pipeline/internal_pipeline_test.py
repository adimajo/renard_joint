import renard_joint.relation_extraction.internal_pipeline as pipeline


def test_evaluate():
    pipeline.evaluate("test",
                      pipeline.bert_model,
                      pipeline.ner_model,
                      pipeline.re_model,
                      pipeline.entity_label_map,
                      pipeline.entity_classes,
                      pipeline.relation_label_map,
                      pipeline.relation_classes)


def test_predict():
    pipeline.predict(["Adrien is testing the Data Harvesting prototype"],
                     pipeline.bert_model,
                     pipeline.ner_model,
                     pipeline.re_model,
                     pipeline.entity_label_map,
                     pipeline.relation_label_map,
                     pipeline.relation_label_map,
                     pipeline.relation_classes)
