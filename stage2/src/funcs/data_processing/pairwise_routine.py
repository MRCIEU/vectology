import numpy as np
import pandas as pd
from loguru import logger
from nxontology import NXOntology

ECHO_STEP = 200


def make_pairwise_flat(
    pairwise_data: np.ndarray, ebi_df: pd.DataFrame, efo_node_df: pd.DataFrame
) -> pd.DataFrame:
    ebi_efo_list = ebi_df["full_id"].tolist()
    efo_list = efo_node_df["efo_id"].tolist()
    res_list = []
    for i in range(len(ebi_efo_list)):
        if i % ECHO_STEP == 0:
            logger.info(f"#{i} / {len(ebi_efo_list)}")
        for j in range(i, len(efo_list)):
            score = 1 - pairwise_data[i][j]
            item = {
                "mapping_id": i + 1,
                "manual": ebi_efo_list[i],
                "prediction": efo_list[j],
                "score": score,
            }
            res_list.append(item)
    res_df = pd.DataFrame(res_list)
    return res_df


def make_top100(
    pairwise_flat: pd.DataFrame, efo_nx: NXOntology, ebi_df: pd.DataFrame
) -> pd.DataFrame:
    # first do `filter_paiwise_file`
    top_num = 100
    df = (
        pairwise_flat.sort_values(by=["score"], ascending=False)
        .groupby("mapping_id")
        .head(top_num)
        .drop_duplicates(subset=["mapping_id", "manual", "prediction"])
        .sort_values(by=["mapping_id", "score"], ascending=[True, False])
    )
    # then do `get_top_using_pairwise_file`
    top_res = []
    for idx_i, row_i in ebi_df.iterrows():
        if idx_i % ECHO_STEP == 0:
            logger.info(f"#{idx_i} / {len(ebi_df)}")
        mapping_id = row_i["mapping_id"]
        efo_predictions = df[df["mapping_id"] == mapping_id].head(n=top_num)[
            ["prediction", "score"]
        ]
        for idx_j, row_j in efo_predictions.iterrows():
            manual_efo = row_i["full_id"]
            predicted_efo = row_j["prediction"]
            score = row_j["score"]
            try:
                res = efo_nx.similarity(manual_efo, predicted_efo).results()
                nx_val = res["batet"]
            except:
                nx_val = 0
            top_res.append(
                {
                    "mapping_id": idx_i + 1,
                    "manual": row_i["full_id"],
                    "prediction": predicted_efo,
                    "score": score,
                    "nx": nx_val,
                }
            )
    res_df = pd.DataFrame(top_res)
    return res_df
