"""A pytorch module for multiple relation extraction (Wang et al. 2019)

REFERENCE:
Wang, H., Tan, M., Yu, M., Chang, S., Wang, D., Xu, K., ... & Potdar, S. (2019).
Extracting multiple-relations in one-pass with pre-trained transformers. arXiv preprint arXiv:1902.01030.

SAMPLE USAGE:
# create a model
model = BertForMre(#number_of_relation_classes)

# extract a sentence and create entity masks & labels
docs = conll04_parser.get_docs("train")
extracted_doc = conll04_parser.extract_doc(docs[0])
e1_mask, e2_mask, labels = generate_entity_mask(
    extracted_doc["data_frame"].shape[0],
    extracted_doc["entity_position"],
    extracted_doc["relations"]
)

# train the model
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)
outputs = model(
    torch.tensor([extracted_doc["data_frame"]["token_ids"]]),
    e1_mask=e1_mask,
    e2_mask=e2_mask,
    labels=labels
)
loss = outputs.loss
loss.backward()
optimizer.step()
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from transformers.modeling_outputs import ModelOutput


class MreOutput(ModelOutput):
    """
    Class for outputs of the multiple relation extraction model.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_entities ** 2, num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True``
            is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or
            when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertForMre(nn.Module):
    """A pytorch module for multiple relation extraction (Wang et al. 2019)

    Reference:
    Wang, H., Tan, M., Yu, M., Chang, S., Wang, D., Xu, K., ... & Potdar, S. (2019).
    Extracting multiple-relations in one-pass with pre-trained transformers. arXiv preprint arXiv:1902.01030.
    """

    def __init__(
            self,
            num_labels,
            model_name="bert-base-uncased"
    ):
        super(BertForMre, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.train()  # Set BERT to training mode
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            e1_mask=None,
            e2_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = None
        if e1_mask is not None and e2_mask is not None:
            num_relation = e1_mask.shape[0]
            seq_length = e1_mask.shape[1]
            sequence_output = torch.stack([sequence_output] * num_relation, dim=1)

            e1_mask = torch.reshape(e1_mask, [-1, num_relation, seq_length, 1])
            e1 = torch.mul(sequence_output, e1_mask.float())
            e1 = torch.sum(e1, dim=-2) / torch.clamp(torch.sum(e1_mask.float(), dim=-2), min=1.0)
            e1 = torch.reshape(e1, [-1, self.bert.config.hidden_size])

            e2_mask = torch.reshape(e2_mask, [-1, num_relation, seq_length, 1])
            e2 = torch.mul(sequence_output, e2_mask.float())
            e2 = torch.sum(e2, dim=-2) / torch.clamp(torch.sum(e2_mask.float(), dim=-2), min=1.0)
            e2 = torch.reshape(e2, [-1, self.bert.config.hidden_size])

            sequence_output = torch.cat([e1, e2], dim=-1)
            logits = self.classifier(sequence_output)

        loss = None
        if logits is not None and labels is not None:
            # print(logits.type(), labels.type())
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:] if logits is not None else outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MreOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def generate_entity_mask(sequence_length, entity_position, relations):
    """For each pair of entities e1 and e2 in a sentence, generate a bit mask for e1 (appended to e1_mask),
    a bit mask for e2 (appended to e2_mask), and append the corresponding relation type of the pair to labels
    """
    relation_count = len(entity_position) * (len(entity_position) - 1)
    e1_mask = torch.zeros((relation_count, sequence_length), dtype=torch.long)
    e2_mask = torch.zeros((relation_count, sequence_length), dtype=torch.long)
    labels = torch.zeros(relation_count, dtype=torch.long)
    i = 0
    for e1 in entity_position:
        for e2 in entity_position:
            if e1 != e2:
                l1, h1 = entity_position[e1]
                l2, h2 = entity_position[e2]
                e1_mask[i, l1:h1] = 1
                e2_mask[i, l2:h2] = 1
                for relation in relations:
                    if relations[relation]["source"] == e1 and relations[relation]["target"] == e2:
                        labels[i] = relations[relation]["type"]
                i += 1
    return e1_mask, e2_mask, labels
