"""
Constants to retrain the model on internal data.

.. autosummary::
    :toctree:

"""
import os

model_path = "bert-base-cased"
model_save_path = os.path.join(os.environ["MODEL"], "spert/internal_")
train_dataset = "Training"
dev_dataset = "Test"
test_dataset = "Test"
entity_types = 9
relation_types = 7
neg_entity_count = 150
neg_relation_count = 200
epochs = 30
patience = 30
lr = 5e-6
lr_warmup = 0.1
weight_decay = 0.01
max_grad_norm = 1.0
relation_filter_threshold = 0.4
width_embedding_size = 25
prop_drop = 0.1
max_span_size = 10
max_pairs = 1000
is_overlapping = False
