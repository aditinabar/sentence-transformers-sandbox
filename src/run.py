import argparse

from models.sentence_transformer import BERTTransformer
from models.mtl_expansion import MultiTaskBERT
from utils.helper import generate_sentence_embeddings
from utils.constants import TRANSFORMER, NUM_CLASSIFICATION_LABELS,NUM_NER_LABELS


def run(args):

    basicBERT = BERTTransformer(TRANSFORMER)
    mtlBERT = MultiTaskBERT(
        model_name=TRANSFORMER,
        num_category_classes=NUM_CLASSIFICATION_LABELS,
        num_ner_entities=NUM_NER_LABELS
    )
    model_map = {
        "backbone": basicBERT,
        "mtl": mtlBERT
    }
    embedding = generate_sentence_embeddings(model_map[args.model_name], args.sentence, task=args.task)
    print(embedding.shape)
    print(embedding[0][:50])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("model_name", type=str, help="options are 'backbone' or 'mtl'")
    parser.add_argument("--task", type=str, help="options are 'ner' or 'classification'")
    parser.add_argument("sentence", type=str, help="Pass in a sentence surrounded by quotes")

    args = parser.parse_args()
    print(args)
    print(args.model_name)

    run(args)

