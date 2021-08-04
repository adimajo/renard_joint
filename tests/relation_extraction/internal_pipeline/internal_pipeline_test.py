from renard_joint.relation_extraction import internal_pipeline


def test_evaluate_internal_pipeline():
    internal_pipeline.evaluate("test",
                               internal_pipeline.bert_model,
                               internal_pipeline.ner_model,
                               internal_pipeline.re_model,
                               internal_pipeline.label_map_bio,
                               internal_pipeline.entity_label_map,
                               internal_pipeline.entity_classes,
                               internal_pipeline.relation_label_map,
                               internal_pipeline.relation_classes)


def test_predict_internal_pipeline():
    internal_pipeline.predict(["Adrien is testing the Data Harvesting prototype"],
                              internal_pipeline.bert_model,
                              internal_pipeline.ner_model,
                              internal_pipeline.re_model,
                              internal_pipeline.label_map_bio,
                              internal_pipeline.entity_label_map,
                              internal_pipeline.relation_label_map)
