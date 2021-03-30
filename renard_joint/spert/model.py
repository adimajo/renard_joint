import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, relation_types: int, entity_types: int,
                 width_embedding_size: int, prop_drop: float, freeze_transformer: bool, max_pairs: int,
                 is_overlapping: bool, relation_filter_threshold: float, relation_possibility: map = None):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers
        self.relation_classifier = nn.Linear(config.hidden_size * 3 + width_embedding_size * 2, relation_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + width_embedding_size, entity_types)
        self.width_embedding = nn.Embedding(100, width_embedding_size)
        self.dropout = nn.Dropout(prop_drop)

        self._hidden_size = config.hidden_size
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._relation_filter_threshold = relation_filter_threshold
        self._relation_possibility = relation_possibility
        self._max_pairs = max_pairs
        self._is_overlapping = is_overlapping  # whether overlapping entities are allowed

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _classify_entity(self, token_embedding, width_embedding, cls_embedding, entity_mask, entity_label):
        """
        INPUT:
        token_embedding.shape = (sentence_length, hidden_size)
        width_embedding.shape = (entity_count, width_embedding_size)
        cls_embedding.shape = (1, hidden_size)
        entity_mask.shape = (entity_count, sentence_length)
        entity_label.shape = (entity_count,)

        RETURN:
        entity_logit.shape = (entity_count, self._entity_types)
        entity_loss -> scala
        entity_pred.shape = (entity_count,)
        """
        sentence_length = token_embedding.shape[0]
        hidden_size = token_embedding.shape[1]
        entity_count = entity_mask.shape[0]

        entity_embedding = token_embedding.view(1, sentence_length, hidden_size) + \
            ((entity_mask == 0) * (-1e30)).view(entity_count, sentence_length, 1)
        entity_embedding = entity_embedding.max(dim=-2)[0]  # maxpool
        # print(entity_embedding)

        entity_embedding = torch.cat([cls_embedding.repeat(entity_count, 1),
                                      entity_embedding,
                                      width_embedding], dim=1)
        entity_embedding = self.dropout(entity_embedding)

        entity_logit = self.entity_classifier(entity_embedding)
        entity_loss = None
        if entity_label is not None:
            # If entity labels are provided, calculate cross entropy loss and take the average over all samples
            # Refer to the paper
            loss_fct = CrossEntropyLoss(reduction='none')
            entity_loss = loss_fct(entity_logit, entity_label)
            entity_loss = entity_loss.sum() / entity_loss.shape[-1]
            # print(entity_loss)

        entity_confidence, entity_pred = F.softmax(entity_logit, dim=-1).max(dim=-1)
        return entity_logit, entity_loss, entity_confidence, entity_pred

    def _filter_span(self, entity_mask: torch.tensor, entity_pred: torch.tensor, entity_confidence: torch.tensor):
        entity_count = entity_mask.shape[0]
        sentence_length = entity_mask.shape[1]
        entities = [(entity_mask[i], entity_pred[i].item(), entity_confidence[i].item()) for i in range(entity_count)]
        entities = sorted(entities, key=lambda entity: entity[2], reverse=True)

        entity_span = []
        entity_embedding = torch.zeros((sentence_length,)) if not self._is_overlapping else None
        entity_type_map = {}

        for i in range(entity_count):
            e_mask, e_pred, e_confidence = entities[i]
            begin = torch.argmax(e_mask).item()
            end = sentence_length - torch.argmax(e_mask.flip(0)).item()

            assert end > begin
            assert e_mask[begin:end].sum() == end - begin
            assert e_mask.sum() == end - begin

            entity_type_map[(begin, end)] = e_pred

            if e_pred != 0:
                if self._is_overlapping:
                    entity_span.append((begin, end, e_pred))
                elif not self._is_overlapping and entity_embedding[begin:end].sum() == 0:
                    entity_span.append((begin, end, e_pred))
                    entity_embedding[begin:end] = e_pred

        return entity_span, entity_embedding, entity_type_map

    def _generate_relation_mask(self, entity_span, sentence_length):
        relation_mask = []
        relation_possibility = []
        for e1 in entity_span:
            for e2 in entity_span:
                if e1 != e2:
                    c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
                    template = [1] * sentence_length
                    template[e1[0]: e1[1]] = [x * 2 for x in template[e1[0]: e1[1]]]
                    template[e2[0]: e2[1]] = [x * 3 for x in template[e2[0]: e2[1]]]
                    template[c[0]: c[1]] = [x * 5 for x in template[c[0]: c[1]]]
                    relation_mask.append(template)
                    if self._relation_possibility is not None:
                        if (e1[2], e2[2]) in self._relation_possibility:
                            relation_possibility.append(self._relation_possibility[(e1[2], e2[2])])
                        else:
                            relation_mask.pop()  # no relation possible, remove the last mask
        return torch.tensor(relation_mask, dtype=torch.long).to(self.device), \
            torch.tensor(relation_possibility, dtype=torch.long).to(self.device)

    def _classify_relation(self, token_embedding, e1_width_embedding, e2_width_embedding,
                           relation_mask, relation_label, relation_possibility):
        """
        INPUT:
        token_embedding.shape = (sentence_length, hidden_size)
        e1_width_embedding.shape = (relation_count, width_embedding_size)
        e2_width_embedding.shape = (relation_count, width_embedding_size)
        relation_mask.shape = (relation_count, sentence_length)
        relation_label.shape = (relation_count,)

        RETURN:
        relation_logit.shape = (relation_count, self._relation_types)
        relation_loss -> scala
        relation_pred.shape = (relation_count,)
        """
        sentence_length = token_embedding.shape[0]
        hidden_size = token_embedding.shape[1]
        relation_count = relation_mask.shape[0]

        e1_embedding = token_embedding.view(1, sentence_length, hidden_size) + \
            ((relation_mask % 2 != 0) * (-1e30)).view(relation_count, sentence_length, 1)
        e1_embedding = e1_embedding.max(dim=-2)[0]  # maxpool
        # print(e1_embedding)

        e2_embedding = token_embedding.view(1, sentence_length, hidden_size) + \
            ((relation_mask % 3 != 0) * (-1e30)).view(relation_count, sentence_length, 1)
        e2_embedding = e2_embedding.max(dim=-2)[0]  # maxpool
        # print(e2_embedding)

        c_embedding = token_embedding.view(1, sentence_length, hidden_size) + \
            ((relation_mask % 5 != 0) * (-1e30)).view(relation_count, sentence_length, 1)
        c_embedding = c_embedding.max(dim=-2)[0]  # maxpool
        c_embedding[c_embedding < -1e15] = 0  # if context is empty then set it to 0
        # print(c_embedding)

        relation_embedding = torch.cat([c_embedding,
                                        e1_embedding, e2_embedding,
                                        e1_width_embedding, e2_width_embedding], dim=1)
        relation_embedding = self.dropout(relation_embedding)

        relation_logit = self.relation_classifier(relation_embedding)
        relation_loss = None
        if relation_label is not None:
            # If relation labels are provided, calculate the binary cross entropy loss
            # and take the sum over all samples
            loss_fct = BCEWithLogitsLoss(reduction='none')
            onehot_relation_label = F.one_hot(relation_label, num_classes=self._relation_types + 1).float()
            onehot_relation_label = onehot_relation_label[::, 1:]  # Ignore None relation class
            relation_loss = loss_fct(relation_logit, onehot_relation_label)
            relation_loss = relation_loss.sum(dim=-1) / relation_loss.shape[-1]
            relation_loss = relation_loss.sum()  # sum to be divided for average later
            # print(relation_loss)

        relation_sigmoid = torch.sigmoid(relation_logit)
        # Filter out low confident relations
        relation_sigmoid[relation_sigmoid < self._relation_filter_threshold] = 0
        relation_sigmoid = torch.cat([torch.zeros((relation_sigmoid.shape[0], 1)).to(self.device), relation_sigmoid],
                                     dim=-1)
        # filter out illogical relations
        if self._relation_possibility is not None and \
                relation_possibility is not None and \
                not torch.equal(relation_possibility, torch.tensor([], dtype=torch.long).to(self.device)):
            relation_sigmoid = torch.mul(relation_sigmoid, relation_possibility)
        relation_confidence, relation_pred = relation_sigmoid.max(dim=-1)

        return relation_logit, relation_loss, relation_confidence, relation_pred

    def _filter_relation(self, relation_mask: torch.tensor, relation_pred: torch.tensor, entity_type_map):
        relation_count = relation_mask.shape[0]
        sentence_length = relation_mask.shape[1]
        relation_span = []

        for i in range(relation_count):
            if relation_pred[i] != 0:
                e1_begin = torch.argmax((relation_mask[i] % 2 == 0).long()).item()
                e1_end = sentence_length - torch.argmax((relation_mask[i].flip(0) % 2 == 0).long()).item()
                assert e1_end > e1_begin
                assert (relation_mask[i, e1_begin:e1_end] % 2).sum() == 0

                e2_begin = torch.argmax((relation_mask[i] % 3 == 0).long()).item()
                e2_end = sentence_length - torch.argmax((relation_mask[i].flip(0) % 3 == 0).long()).item()
                assert e2_end > e2_begin
                assert (relation_mask[i, e2_begin:e2_end] % 3).sum() == 0
                # print(e1_begin, e1_end, e2_begin, e2_end)

                relation_span.append(((e1_begin, e1_end, entity_type_map[(e1_begin, e1_end)]),
                                      (e2_begin, e2_end, entity_type_map[(e2_begin, e2_end)]),
                                      relation_pred[i].item()))

        return relation_span

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids: torch.tensor,
                entity_mask: torch.tensor = None, entity_label: torch.tensor = None,
                relation_mask: torch.tensor = None, relation_label: torch.tensor = None,
                is_training: bool = True):
        # get the last hidden layer from BERT
        bert_embedding = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)['last_hidden_state']

        # get the CLS and other tokens embedding
        bert_embedding = torch.reshape(bert_embedding, (-1, self._hidden_size))
        cls_embedding = bert_embedding[:1]  # CLS is the first element
        token_embedding = bert_embedding[1:-1]  # everything except CLS and SEP at both ends

        # get the width embedding for each entity length
        width_embedding = self.width_embedding(torch.sum(entity_mask, dim=-1))
        entity_logit, entity_loss, entity_confidence, entity_pred \
            = self._classify_entity(token_embedding, width_embedding, cls_embedding, entity_mask, entity_label)

        entity_span, entity_embedding, entity_type_map = self._filter_span(entity_mask, entity_pred, entity_confidence)

        # if not relation_mask then generate them from pairs of entities
        # only for prediction and evaluation
        relation_possibility = None
        if not is_training or relation_mask is None:
            relation_mask, relation_possibility = self._generate_relation_mask(entity_span, token_embedding.shape[0])
            relation_label = None

        # return immediately if there is no relation to predict (e.g. there are less than 2 entities)
        output = {
            "loss": entity_loss,
            "entity": {
                "logit": entity_logit,
                "loss": None if entity_loss is None else entity_loss.item(),
                "pred": entity_pred,
                "confidence": entity_confidence,
                "span": entity_span,
                "embedding": entity_embedding
            },
            "relation": None
        }
        if relation_mask is None or torch.equal(relation_mask, torch.tensor([], dtype=torch.long).to(self.device)):
            return output

        relation_count = relation_mask.shape[0]
        relation_logit = torch.zeros((relation_count, self._relation_types))
        relation_loss = []
        relation_confidence = torch.zeros((relation_count,))
        relation_pred = torch.zeros((relation_count,), dtype=torch.long)
        e1_width_embedding = self.width_embedding(torch.sum(relation_mask % 2 == 0, dim=-1))
        e2_width_embedding = self.width_embedding(torch.sum(relation_mask % 3 == 0, dim=-1))

        # break down relation_mask (list of possible relations) to smaller chunks
        for i in range(0, relation_count, self._max_pairs):
            j = min(relation_count, i + self._max_pairs)
            logit, loss, confidence, pred = self._classify_relation(token_embedding,
                                                                    e1_width_embedding[i: j],
                                                                    e2_width_embedding[i: j],
                                                                    relation_mask[i: j],
                                                                    relation_label[i: j] if relation_label is not None else None,
                                                                    relation_possibility[i: j] if relation_possibility is not None else None)
            relation_logit[i: j] = logit
            if loss is not None:
                relation_loss.append(loss)
            relation_confidence[i: j] = confidence
            relation_pred[i: j] = pred

        # relation loss is the average of binary cross entropy loss of each sample
        # refer to the paper
        relation_loss = None if len(relation_loss) == 0 else (sum(relation_loss) / relation_count)
        relation_span = self._filter_relation(relation_mask, relation_pred, entity_type_map)

        # Final loss is the sum of entity_loss and relation_loss
        if relation_loss is not None:
            output["loss"] = relation_loss + entity_loss
        output["relation"] = {
            "logit": relation_logit,
            "loss": None if relation_loss is None else relation_loss.item(),
            "pred": relation_pred,
            "confidence": relation_confidence,
            "span": relation_span
        }
        return output
