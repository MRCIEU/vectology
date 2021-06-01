# general params
SEED: int = 42
NUM_WORKERS: int = 2
LOGGER_STEP: int = 500
MODEL_NAME: str = "bert-base-uncased"
MNLI_MODEL_NAME: str = "bert-base-uncased"
WIKITEXT_MODEL_NAME: str = "gsarti/biobert-nli"
EFO_LM_MODEL_NAME: str = "gsarti/biobert-nli"
# EFO_CLS_MODEL_NAME: str = "gsarti/biobert-nli"
EFO_CLS_MODEL_NAME: str = "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"

# data processing params
MAX_TOKENIZATION_LENGTH: int = 128
BATCH_SIZE_GENERAL: int = 64
# typically for language modelling the batch size needs to be
# much smaller
BATCH_SIZE_LM: int = 32
MLM_PROBABILITY: float = 0.15

# model architecture params
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 0.01
ADAM_EPSILON: float = 1e-8

# trainer params
NUM_TRAIN_EPOCHS: int = 5
