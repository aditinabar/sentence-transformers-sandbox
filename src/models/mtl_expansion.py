import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name, num_category_classes, num_ner_entities):
        super(MultiTaskBERT, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.sentence_classifier_head = nn.Linear(self.bert.config.hidden_size, num_category_classes)
        self.ner_head = nn.Linear(self.bert.config.hidden_size, num_ner_entities)

    def forward(self, input_ids, attention_mask, task):
        # model is initialized without decoder
        # get the cls token embedding from the last hidden state, basically the sentence embeddings from encoder.
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # use mean pooling of the hidden state embeddings instead of the cls token embedding given by output.pooler_output
        if task == "classification":
            pooled_sentence_embedding = self._mean_pooling(output.last_hidden_state, attention_mask)
            task_output = self.sentence_classifier_head(pooled_sentence_embedding)

        elif task == "ner":
            task_output = self.ner_head(output.last_hidden_state)
        
        return task_output

    def _mean_pooling(self, hidden_states, attention_mask):
        # expand the mask to the dimensions of the hidden states
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())

        # sum embeddings along the dimension of the tokens so that our output remains a vector
        sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
        sum_mask = expanded_mask.sum(dim=1)
        return sum_embeddings / sum_mask
