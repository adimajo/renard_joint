"""
Constants to retrain the model on SciERC data.

.. autosummary::
    :toctree:

"""
import os

model_path = "bert-base-cased"
model_save_path = os.path.join(os.environ["MODEL"], "spert/scierc_")
train_dataset = "train"
dev_dataset = "dev"
test_dataset = "test"
entity_types = 7
relation_types = 7
neg_entity_count = 700
neg_relation_count = 100
epochs = 20
lr = 5e-5
lr_warmup = 0.1
weight_decay = 0.01
max_grad_norm = 1.0
relation_filter_threshold = 0.8
width_embedding_size = 25
prop_drop = 0.1
max_span_size = 10
max_pairs = 1000
is_overlapping = True
