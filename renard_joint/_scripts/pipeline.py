"""
NER-RE pipeline script.
"""
import sys


def main():
    if len(sys.argv) <= 2:
        raise ValueError("No argument found, usage: python pipeline.py [dataset] [evaluate/predict] [sentences]")
    elif sys.argv[1] == "conll04":
        from renard_joint.relation_extraction import conll04_pipeline as le_pipeline
    elif sys.argv[1] == "internal":
        from renard_joint.relation_extraction import internal_pipeline as le_pipeline
    else:
        raise ValueError("Invalid dataset argument: only 'conll04' and 'internal' are supported.")

    if sys.argv[2] == "evaluate":
        le_pipeline.evaluate("Test",
                             le_pipeline.bert_model,
                             le_pipeline.ner_model,
                             le_pipeline.re_model,
                             le_pipeline.label_map_bio,
                             le_pipeline.entity_label_map,
                             le_pipeline.entity_classes,
                             le_pipeline.relation_label_map,
                             le_pipeline.relation_classes)
    elif sys.argv[2] == "predict":
        le_pipeline.predict(sys.argv[3:],
                            le_pipeline.bert_model,
                            le_pipeline.ner_model,
                            le_pipeline.re_model,
                            le_pipeline.label_map_bio,
                            le_pipeline.entity_label_map,
                            le_pipeline.relation_label_map)
    else:
        raise ValueError("Invalid argument(s)")


if __name__ == "__main__":
    main()
