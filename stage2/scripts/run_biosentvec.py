import numpy as np
import pandas as pd
import sent2vec
from metaflow import FlowSpec, Parameter, step
from scipy.spatial import distance

from funcs import paths
from funcs.data_processing import mapping_routine, pairwise_routine, stage1_processing
from funcs.nlp import nlp

from icecream import ic  # noqa


ECHO_STEP = 200

BIOSENTVEC_MODEL_PATH = paths.init["biosentvec_model"]
assert BIOSENTVEC_MODEL_PATH.exists()

OUTPUT_PATH = paths.stage2["output"]
assert OUTPUT_PATH.exists(), OUTPUT_PATH


class BiosentvecFlow(FlowSpec):

    OVERWRITE = Parameter(
        "overwrite",
        help="overwrite",
        default=False,
        is_flag=True,
    )

    @step
    def start(self):
        "Init."

        self.ebi_df = stage1_processing.get_ebi_data()
        self.efo_node_df = stage1_processing.get_efo_nodes()
        self.efo_nx = stage1_processing.get_efo_nx()

        self.next(self.make_encode)

    @step
    def make_encode(self):
        ebi_encode_path = OUTPUT_PATH / "biosentvec-ebi-encode.npy"
        efo_node_encode_path = OUTPUT_PATH / "biosentvec-efo-encode.npy"
        if (
            not ebi_encode_path.exists()
            or not efo_node_encode_path.exists()
            or self.OVERWRITE
        ):
            self.ebi_encode, self.efo_encode = self._make_encode()
            np.save(str(ebi_encode_path), self.ebi_encode)
            np.save(str(efo_node_encode_path), self.efo_encode)
        else:
            print("Read from cache")
            self.ebi_encode = np.load(str(ebi_encode_path))
            self.efo_encode = np.load(str(efo_node_encode_path))
        self.next(self.make_pairwise)

    @step
    def make_pairwise(self):
        pairwise_results_path = OUTPUT_PATH / "biosentvec-dd.npy"
        if not pairwise_results_path.exists() or self.OVERWRITE:
            self.pairwise_results = distance.cdist(
                self.ebi_encode, self.efo_encode, "cosine"
            )
            np.save(str(pairwise_results_path), self.pairwise_results)
        else:
            self.pairwise_results = np.load(pairwise_results_path)
        self.next(self.make_pairwise_flat)

    @step
    def make_pairwise_flat(self):
        pairwise_flat_path = OUTPUT_PATH / "biosentvec-pairwise.csv.gz"
        if not pairwise_flat_path.exists() or self.OVERWRITE:
            self.pairwise_flat = pairwise_routine.make_pairwise_flat(
                pairwise_data=self.pairwise_results,
                ebi_df=self.ebi_df,
                efo_node_df=self.efo_node_df,
            )
            self.pairwise_flat.to_csv(pairwise_flat_path, index=False)
        else:
            self.pairwise_flat = pd.read_csv(pairwise_flat_path)
        self.next(self.make_top100)

    @step
    def make_top100(self):
        top100_path = OUTPUT_PATH / "biosentvec-top-100.csv"
        if not top100_path.exists() or self.OVERWRITE:
            self.top100_results = pairwise_routine.make_top100(
                pairwise_flat=self.pairwise_flat, efo_nx=self.efo_nx, ebi_df=self.ebi_df
            )
            self.top100_results.to_csv(top100_path, index=False)
        else:
            self.top100_results = pd.read_csv(top100_path)
        self.next(self.verify)

    @step
    def verify(self):

        model_collection = {
            "BioSentVec": {
                "model": "BioSentVec",
                "stage": "stage2",
                "top_100": paths.stage1["output2_dir"] / "BioSentVec-top-100.tsv.gz",
                "pairwise_filter": paths.stage1["output2_dir"]
                / "BioSentVec-pairwise-filter.tsv.gz",
                "top_100_revised": paths.stage2["output"] / "biosentvec-top-100.csv",
            },
        }

        trait_efo_mapping_df = mapping_routine.prep_trait_efo_mapping_agg(
            ebi_data=self.ebi_df, model_collection=model_collection, batet_score=1.0
        )
        print(trait_efo_mapping_df)

        self.next(self.end)

    @step
    def end(self):
        "Summary and done."

    def _make_encode(self):
        print("Init biosentvec model")
        biosentvec_model = sent2vec.Sent2vecModel()
        biosentvec_model.load_model(str(BIOSENTVEC_MODEL_PATH))

        print("Encode ebi")
        ebi_encode = nlp.biosentvec_encode_terms(
            text_list=self.ebi_df["query"].tolist(),
            biosentvec_model=biosentvec_model,
        )
        ic(ebi_encode.shape)

        print("Encode efo")
        efo_encode = nlp.biosentvec_encode_terms(
            text_list=self.efo_node_df["efo_label"].tolist(),
            biosentvec_model=biosentvec_model,
        )
        ic(efo_encode.shape)

        return ebi_encode, efo_encode


if __name__ == "__main__":
    BiosentvecFlow()
