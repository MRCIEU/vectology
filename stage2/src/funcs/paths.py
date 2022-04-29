from pathlib import Path
from typing import Dict

from . import utils

_project_root = utils.find_project_root()
_data_root = utils.find_data_root()
_model_root = _project_root / "models"

# Init
_stage1_dir = _data_root / "vectology-stage1" / "paper"
assert _stage1_dir.exists()
init: Dict[str, Path] = {
    "efo_nodes": _stage1_dir / "input" / "efo_nodes_2021_02_01.csv",
    "efo_edges": _stage1_dir / "input" / "efo_edges_2021_02_01.csv",
    "ukb_master": _stage1_dir / "input" / "UK_Biobank_master_file.tsv",
    "biosentvec_model": _model_root
    / "biosentvec"
    / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin",
    "bioconceptvec_skipgram": _model_root
    / "bioconceptvec"
    / "bioconceptvec_word2vec_skipgram.bin",
    "scispacy_md": _model_root
    / "en_core_sci_md-0.5.0"
    / "en_core_sci_md"
    / "en_core_sci_md-0.5.0",
}
for k, v in init.items():
    assert v.exists()

# stage1, strictly immutable
_stage1_manuscript_dir = _data_root / "vectology-manuscript"
_stage1_output1_dir = _stage1_dir / "output" / "trait-trait-v1-lowercase"
_stage1_output2_dir = _stage1_dir / "output" / "trait-efo-v2-lowercase"
stage1: Dict[str, Path] = {
    "ebi_ukb_cleaned": _stage1_manuscript_dir / "manuscript" / "s3_ebi-ukb-cleaned.csv",
    "output1_dir": _stage1_output1_dir,
    "output2_dir": _stage1_output2_dir,
    "stage1_wa_nx_1_all": _stage1_output2_dir / "weighted-average-nx-1-all.csv",
    "stage1_wa_nx_10_all": _stage1_output2_dir / "weighted-average-nx-10-all.csv",
    "stage1_com_scores": _stage1_output1_dir / "com_scores.tsv.gz",
}
for k, v in stage1.items():
    assert v.exists()

# stage2
_stage2_dir = _data_root / "output"
assert _stage2_dir.exists()
_stage2_revised_dir = _stage2_dir / "stage1-assets-revised"
_stage2_revised_dir.mkdir(exist_ok=True)
_stage2_stage1_cache_dir = _stage2_dir / "stage1-cache"
_stage2_stage1_cache_dir.mkdir(exist_ok=True)
_stage2_output_dir = _stage2_dir / "stage2-output"
_stage2_output_dir.mkdir(exist_ok=True)
_stage2_tmp_dir = _stage2_dir / "tmp"
_stage2_tmp_dir.mkdir(exist_ok=True)
stage2: Dict[str, Path] = {
    "revised": _stage2_revised_dir,
    "stage1-cache": _stage2_stage1_cache_dir,
    "output": _stage2_output_dir,
    "tmp": _stage2_tmp_dir,
    "efo_nx": _stage2_stage1_cache_dir / "efo_nx.gpickle",
}

# final artifacts
_analysis_artifacts_dir = utils.find_analysis_artifacts_dir()
analysis_artifacts: Dict[str, Path] = {
    "dir": _analysis_artifacts_dir,
}
