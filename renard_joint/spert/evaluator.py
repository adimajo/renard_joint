import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def evaluate_results(true_labels, predicted_labels, label_map, classes):
    """Compare the prediction list with the truth values"""
    precision, recall, fbeta_score, support \
        = precision_recall_fscore_support(true_labels, predicted_labels, average=None, labels=classes,
                                          zero_division=0)
    result = pd.DataFrame(index=[label_map[c] for c in classes])
    result["precision"] = precision
    result["recall"] = recall
    result["fbeta_score"] = fbeta_score
    result["support"] = support
    result.loc["macro"] = list(precision_recall_fscore_support(true_labels, predicted_labels, average="macro",
                                                               labels=classes, zero_division=0))
    result.loc["micro"] = list(precision_recall_fscore_support(true_labels, predicted_labels, average="micro",
                                                               labels=classes, zero_division=0))
    return result


def evaluate_span(true_span, pred_span, label_map, classes):
    """Evaluate 2 set of spans
    An entity is considered correct if its predicted begin, end, and type are all correct
    A relation is considered correct if both of its entities (including begins, ends, and entity types)
    are correct and the relation type is correct
    """
    assert len(true_span) == len(pred_span)

    true_label = []
    pred_label = []

    for true_span_batch, pred_span_batch in zip(true_span, pred_span):
        true_span_batch = dict([((item[0][:2] if isinstance(item[0], tuple) else item[0],
                                  item[1][:2] if isinstance(item[1], tuple) else item[1]),
                                 item[2]) for item in true_span_batch])
        pred_span_batch = dict([((item[0][:2] if isinstance(item[0], tuple) else item[0],
                                  item[1][:2] if isinstance(item[1], tuple) else item[1]),
                                 item[2]) for item in pred_span_batch])
        s = set()
        s.update(true_span_batch.keys())
        s.update(pred_span_batch.keys())
        for span in s:
            if span in true_span_batch:
                true_label.append(true_span_batch[span])
            else:
                true_label.append(0)
            if span in pred_span_batch:
                pred_label.append(pred_span_batch[span])
            else:
                pred_label.append(0)

    assert len(true_label) == len(pred_label)
    return evaluate_results(true_label, pred_label, label_map, classes)


def evaluate_loose_relation_span(true_relation_span, pred_relation_span, label_map, classes):
    """Evaluate the relation spans loosely
    A relation is considered correct if its entities has at least a token intersecting
    with the truth values and its type is correct
    """
    assert len(true_relation_span) == len(pred_relation_span)
    approx_relation_span = []

    for true_relation, pred_relation in zip(true_relation_span, pred_relation_span):
        approx_relation = []
        for pr in pred_relation:
            found = False
            pr_e1 = pr[0]
            pr_e2 = pr[1]
            for tr in true_relation:
                tr_e1 = tr[0]
                tr_e2 = tr[1]
                if (tr_e1[0] <= pr_e1[0] < tr_e1[1] or tr_e1[0] < pr_e1[1] <= tr_e1[1]) and \
                        (tr_e2[0] <= pr_e2[0] < tr_e2[1] or tr_e2[0] < pr_e2[1] <= tr_e2[1]) and \
                        tr[2] == pr[2]:
                    found = True
                    # if there is a true relation that approximately match the predicted relation
                    # then use that true relation as the predicted value
                    approx_relation.append(tr)
                    break
            if not found:
                # else use the original predicted value
                approx_relation.append(pr)

        assert len(approx_relation) == len(pred_relation)
        approx_relation_span.append(approx_relation)

    # run the strict evaluation with the approximated relations
    return evaluate_span(true_relation_span, approx_relation_span, label_map, classes)
