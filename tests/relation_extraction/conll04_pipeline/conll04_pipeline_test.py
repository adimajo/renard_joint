import pytest
from renard_joint.relation_extraction import conll04_pipeline


@pytest.mark.xfail(reason="Not on Gitlab")
def test_evaluate_conll_pipeline():
    conll04_pipeline.evaluate("test",
                              conll04_pipeline.bert_model,
                              conll04_pipeline.ner_model,
                              conll04_pipeline.re_model,
                              conll04_pipeline.label_map_bio,
                              conll04_pipeline.entity_label_map,
                              conll04_pipeline.entity_classes,
                              conll04_pipeline.relation_label_map,
                              conll04_pipeline.relation_classes)


@pytest.mark.xfail(reason="Not on Gitlab")
def test_predict_conll_pipeline():
    from renard_joint.relation_extraction import conll04_pipeline
    conll04_pipeline.predict(["Adrien is testing the Data Harvesting prototype"],
                             conll04_pipeline.bert_model,
                             conll04_pipeline.ner_model,
                             conll04_pipeline.re_model,
                             conll04_pipeline.label_map_bio,
                             conll04_pipeline.entity_label_map,
                             conll04_pipeline.relation_label_map)
