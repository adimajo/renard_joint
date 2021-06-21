import os

import pytest

if os.environ.get("GITLAB", 0) == 1:
    import renard_joint.relation_extraction.conll04_pipeline as pipeline


@pytest.mark.xfail(not os.environ.get("GITLAB", 0) == 1)
def test_evaluate_conll_pipeline():
    pipeline.evaluate("test",
                      pipeline.bert_model,
                      pipeline.ner_model,
                      pipeline.re_model,
                      pipeline.entity_label_map,
                      pipeline.entity_classes,
                      pipeline.relation_label_map,
                      pipeline.relation_classes)


@pytest.mark.xfail(not os.environ.get("GITLAB", 0) == 1)
def test_predict_conll_pipeline():
    pipeline.predict(["Adrien is testing the Data Harvesting prototype"],
                     pipeline.bert_model,
                     pipeline.ner_model,
                     pipeline.re_model,
                     pipeline.entity_label_map,
                     pipeline.relation_label_map)
