"""
NER-RE pipeline script.
"""
import sys
from renard_joint.relation_extraction import conll04_pipeline, internal_pipeline


def main(argv):
    if len(argv) <= 2:
        raise ValueError("No argument found, usage: python pipeline.py [dataset] [evaluate/predict] [sentences]")
    elif argv[1] == "conll04":
        le_pipeline = conll04_pipeline
    elif argv[1] == "internal":
        le_pipeline = internal_pipeline
    else:
        raise ValueError("Invalid dataset argument: only 'conll04' and 'internal' are supported.")

    if argv[2] == "evaluate":
        le_pipeline.evaluate("Test",
                             le_pipeline.bert_model,
                             le_pipeline.ner_model,
                             le_pipeline.re_model,
                             le_pipeline.label_map_bio,
                             le_pipeline.entity_label_map,
                             le_pipeline.entity_classes,
                             le_pipeline.relation_label_map,
                             le_pipeline.relation_classes)
    elif argv[2] == "predict":
        le_pipeline.predict(argv[3:],
                            le_pipeline.bert_model,
                            le_pipeline.ner_model,
                            le_pipeline.re_model,
                            le_pipeline.label_map_bio,
                            le_pipeline.entity_label_map,
                            le_pipeline.relation_label_map)
    else:
        raise ValueError("Invalid argument(s)")


if __name__ == "__main__":
    main(argv=sys.argv)
