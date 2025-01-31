import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

class BERTTransformer(nn.Module):
    def __init__(self, model_name):
        super(BERTTransformer, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, **kwargs):
        # model is initialized without decoder
        # get the cls token embedding from the last hidden state, basically the sentence embeddings from encoder.
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = output.last_hidden_state[:, 0, :]
        
        return embedding
