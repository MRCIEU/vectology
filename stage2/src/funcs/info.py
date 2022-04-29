from pathlib import Path
from typing import Dict

from typing_extensions import TypedDict

from . import paths

revised_cache = paths.stage2["stage1-cache"]
orig_cache = paths.stage1["output2_dir"]


class ModelInfo(TypedDict):
    model: str
    pairwise_filter: Path
    top_100: Path


model_collection: Dict[str, ModelInfo] = {
    "BLUEBERT-EFO": {
        "model": "BLUEBERT-EFO",
        "pairwise_filter": orig_cache / "BLUEBERT-EFO-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "BLUEBERT-EFO-top-100.tsv.gz",
    },
    "BioBERT": {
        "model": "biobert_v1.1_pubmed",
        "pairwise_filter": orig_cache / "BioBERT-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "BioBERT-top-100.tsv.gz",
    },
    "BioSentVec": {
        "model": "BioSentVec",
        "pairwise_filter": orig_cache / "BioSentVec-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "BioSentVec-top-100.tsv.gz",
    },
    "BlueBERT": {
        "model": "NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
        "pairwise_filter": orig_cache / "BlueBERT-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "BlueBERT-top-100.tsv.gz",
    },
    "GUSE": {
        "model": "GUSEv4",
        "pairwise_filter": orig_cache / "GUSE-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "GUSE-top-100.tsv.gz",
    },
    "Spacy": {
        "model": "en_core_web_lg",
        "pairwise_filter": orig_cache / "Spacy-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "Spacy-top-100.tsv.gz",
    },
    "SciSpacy": {
        "model": "en_core_sci_lg",
        "pairwise_filter": orig_cache / "SciSpacy-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "SciSpacy-top-100.tsv.gz",
    },
    "Zooma": {
        "model": "Zooma",
        "pairwise_filter": orig_cache / "Zooma-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "Zooma-top-100.tsv.gz",
    },
    "Levenshtein": {
        "model": "Levenshtein",
        "pairwise_filter": orig_cache / "Levenshtein-pairwise-filter.tsv.gz",
        "top_100": orig_cache / "Levenshtein-top-100.tsv.gz",
    },
}
for k, v in model_collection.items():
    assert v["top_100"].exists()
    assert v["pairwise_filter"].exists()
