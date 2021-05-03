"""
NER-RE pipeline script.
"""
import sys
from . import conll04_pipeline
from . import internal_pipeline


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("No argument found")
    elif sys.argv[1] == "conll04":
        if sys.argv[2] == "evaluate":
            conll04_pipeline.evaluate("test",
                                      conll04_pipeline.bert_model,
                                      conll04_pipeline.ner_model,
                                      conll04_pipeline.re_model,
                                      conll04_pipeline.entity_label_map,
                                      conll04_pipeline.entity_classes,
                                      conll04_pipeline.relation_label_map,
                                      conll04_pipeline.relation_classes)
        elif sys.argv[1] == "predict":
            conll04_pipeline.predict(sys.argv[3:],
                                     conll04_pipeline.bert_model,
                                     conll04_pipeline.ner_model,
                                     conll04_pipeline.re_model,
                                     conll04_pipeline.entity_label_map,
                                     conll04_pipeline.relation_label_map)
        else:
            raise ValueError("Invalid argument(s)")
    elif sys.argv[1] == "internal":
        if len(sys.argv) <= 1:
            raise ValueError("No argument found")
        elif sys.argv[2] == "evaluate":
            internal_pipeline.evaluate("Test",
                                       internal_pipeline.bert_model,
                                       internal_pipeline.ner_model,
                                       internal_pipeline.re_model,
                                       internal_pipeline.entity_label_map,
                                       internal_pipeline.entity_classes,
                                       internal_pipeline.relation_label_map,
                                       internal_pipeline.relation_classes)
        elif sys.argv[2] == "predict":
            internal_pipeline.predict(sys.argv[3:],
                                      internal_pipeline.bert_model,
                                      internal_pipeline.ner_model,
                                      internal_pipeline.re_model,
                                      internal_pipeline.entity_label_map,
                                      internal_pipeline.entity_classes,
                                      internal_pipeline.relation_label_map,
                                      internal_pipeline.relation_classes)
        else:
            raise ValueError("Invalid argument(s)")
    else:
        raise ValueError("Invalid argument(s)")
