import pandas as pd
from transformers import AutoTokenizer

from utils.constants import TRANSFORMER

# # toy dataset if you want sentences to pass
# annotated_df = pd.read_csv("annotated_w_quotes.csv").rename({"annotations": "ner_annotations"}, axis=1)
# included_cols = ["headline", "short_description"]
# classification_label_col = "category"
# ner_label_col = "ner_annotations"

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER)


def get_input_sentence(row, included_columns=None):
    """Helper to generate sentences at row level on a dataframe, or just returns a sentence"""
    if isinstance(row, pd.Series):
        return ' '.join([row[col] for col in included_columns])
    else:
        return row

def generate_sentence_embeddings(model, input_text, included_columns=None, task=None):
    # build input sentence
    sentence = get_input_sentence(input_text, included_columns)

    # get encoded input sentence
    encoded_input = tokenizer(sentence, padding=True, return_tensors="pt")

    # get bert sentence embedding with contextual understanding
    bert_sent_embedding = model.forward(encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"], task=task)
    return bert_sent_embedding
