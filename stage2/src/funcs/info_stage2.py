from typing import Dict

import seaborn as sns

from . import info, paths

revised_cache = paths.stage2["stage1-cache"]
orig_cache = paths.stage1["output2_dir"]
stage2_cache = paths.stage2["output"]

model_collection: Dict[str, info.ModelInfo] = {
    "BLUEBERT-EFO": {
        "model": "BLUEBERT-EFO",
        "pairwise_filter": orig_cache / "BLUEBERT-EFO-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "BLUEBERT-EFO-top-100.csv",
    },
    "BioBERT": {
        "model": "biobert_v1.1_pubmed",
        "pairwise_filter": orig_cache / "BioBERT-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "BioBERT-top-100.csv",
    },
    "BioSentVec": {
        "model": "BioSentVec",
        "pairwise_filter": orig_cache / "BioSentVec-pairwise-filter.tsv.gz",
        # "top_100": stage2_cache / "biosentvec-top-100.csv",
        "top_100": revised_cache / "BioSentVec-top-100.csv",
    },
    # "BioSentVec-BioConceptVec": {
    #     "model": "BioSentVec-BioConceptVec",
    #     "pairwise_filter": orig_cache
    #     / "biosentve-bioconceptvec-skipgram-filter.tsv.gz",
    #     "top_100": stage2_cache / "biosentvec-bioconceptvec-skipgram-top-100.csv",
    # },
    "BlueBERT": {
        "model": "NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
        "pairwise_filter": orig_cache / "BlueBERT-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "BlueBERT-top-100.csv",
    },
    "GUSE": {
        "model": "GUSEv4",
        "pairwise_filter": orig_cache / "GUSE-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "GUSE-top-100.csv",
    },
    "Spacy": {
        "model": "en_core_web_lg",
        "pairwise_filter": orig_cache / "Spacy-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "Spacy-top-100.tsv",
    },
    "SciSpacy": {
        "model": "en_core_sci_lg",
        "pairwise_filter": orig_cache / "SciSpacy-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "SciSpacy-top-100.csv",
    },
    "Zooma": {
        "model": "Zooma",
        "pairwise_filter": orig_cache / "Zooma-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "Zooma-top-100.csv",
    },
    "Levenshtein": {
        "model": "Levenshtein",
        "pairwise_filter": orig_cache / "Levenshtein-pairwise-filter.tsv.gz",
        "top_100": revised_cache / "Levenshtein-top-100.csv",
    },
}

sns_palette = {
    k: sns.color_palette()[idx] for idx, (k, v) in enumerate(model_collection.items())
}
