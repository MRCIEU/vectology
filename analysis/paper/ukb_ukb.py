import pandas as pd
import numpy as np
import requests
import json
import time
import re
import os
import gzip
import timeit
import Levenshtein
import matplotlib.pyplot as plt
from scripts.vectology_functions import (
    create_aaa_distances,
    create_pair_distances,
    embed_text,
    encode_traits,
    create_efo_nxo,
)
from loguru import logger
from pandas_profiling import ProfileReport
from skbio.stats.distance import mantel
from pathlib import Path

import seaborn as sns

# Apply the default theme
sns.set_theme()

# globals
ebi_data = "data/UK_Biobank_master_file.tsv"
efo_rels_v1 = "data/efo_edges_2021_02_01.csv"
efo_rels_v2 = "data/efo_edges.csv"
nxontology_measure = "batet"
top_x = 100

# define the models and set some colours
cols = sns.color_palette()
modelData = [
    {"name": "BLUEBERT-EFO", "model": "BLUEBERT-EFO", "col": cols[0]},
    {"name": "BioBERT", "model": "biobert_v1.1_pubmed", "col": cols[1]},
    {"name": "BioSentVec", "model": "BioSentVec", "col": cols[2]},
    {
        "name": "BlueBERT",
        "model": "NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
        "col": cols[3],
    },
    {"name": "GUSE", "model": "GUSEv4", "col": cols[4]},
    {"name": "Spacy", "model": "en_core_web_lg", "col": cols[5]},
    {"name": "SciSpacy", "model": "en_core_sci_lg", "col": cols[6]},
    {"name": "Zooma", "model": "Zooma", "col": cols[7]},
    {"name": "Levenshtein", "model": "Levenshtein", "col": cols[8]},
]
palette = {}
for m in modelData:
    palette[m["name"]] = m["col"]

output = "output/trait-trait-v1-lowercase"
Path(output).mkdir(parents=True, exist_ok=True)
Path(f"{output}/images").mkdir(parents=True, exist_ok=True)

# create an nxontology instance for EFO hierarchy
def create_nx():
    # create nxontology network of EFO relationships
    logger.info("Creating nx...")
    efo_rel_df = pd.read_csv(efo_rels_v1)
    efo_nx = create_efo_nxo(
        df=efo_rel_df, child_col="efo.id", parent_col="parent_efo.id"
    )
    efo_nx.freeze()
    return efo_nx

def read_ebi():
    # read cleaned EBI data
    ebi_df = pd.read_csv("output/ebi-ukb-cleaned.csv")
    logger.info(ebi_df.head())
    logger.info(ebi_df.shape)

    # limit to Exact
    ebi_df_dedup = ebi_df[ebi_df["MAPPING_TYPE"] == "Exact"]
    logger.info(ebi_df_dedup.shape)

    # now we need one to one mappings of query and EFO, so drop duplicates
    ebi_df_dedup = ebi_df_dedup.drop_duplicates(subset=["full_id"])
    ebi_df_dedup = ebi_df_dedup.drop_duplicates(subset=["query"])
    logger.info(ebi_df_dedup.shape)
    logger.info(ebi_df_dedup["MAPPING_TYPE"].value_counts())

    ebi_df_dedup.to_csv(f"{output}/ebi_exact.tsv.gz", sep="\t")
    # ebi_df_dedup = ebi_df_dedup.head(n=10)
    return ebi_df, ebi_df_dedup


def create_nx_pairs(ebi_df_dedup, efo_nx):
    # create nx score for each full_id
    logger.info(ebi_df_dedup.shape)
    f = f"{output}/nx-ebi-pairs.tsv.gz"
    if os.path.exists(f):
        logger.info("nx for ebi done")
    else:
        data = []
        counter = 0
        # efos = list(ebi_df_dedup['full_id'])
        for i, row1 in ebi_df_dedup.iterrows():
            m1 = row1["mapping_id"]
            e1 = row1["full_id"]
            q1 = row1["query"]
            if counter % 100 == 0:
                logger.info(counter)
            for j, row2 in ebi_df_dedup.iterrows():
                m2 = row2["mapping_id"]
                e2 = row2["full_id"]
                q2 = row2["query"]
                # if e1 != e2:
                res = similarity = efo_nx.similarity(e1, e2).results()
                nx_val = res[nxontology_measure]
                data.append(
                    {
                        "m1": m1,
                        "m2": m2,
                        "e1": e1,
                        "e2": e2,
                        "q1": q1,
                        "q2": q2,
                        "nx": nx_val,
                    }
                )
            counter += 1
        logger.info(counter)
        df = pd.DataFrame(data)
        df.to_csv(f, sep="\t", index=False)
    logger.info("Done")


def create_nx_pairs_nr(ebi_df, efo_nx):
    # create nx score for each full_id (non-redundant)
    logger.info(ebi_df.shape)
    f = f"{output}/nx-ebi-pairs-nr.tsv.gz"
    if os.path.exists(f):
        logger.info("nx for ebi done")
    else:
        data = []
        counter = 0
        # efos = list(ebi_df_dedup['full_id'])
        # for i,row1 in ebi_df.iterrows():
        for i in range(0, ebi_df.shape[0]):
            m1 = ebi_df.iloc[i]["mapping_id"]
            e1 = ebi_df.iloc[i]["full_id"]
            q1 = ebi_df.iloc[i]["query"]
            if counter % 100 == 0:
                logger.info(counter)
            for j in range(i, ebi_df.shape[0]):
                # for j,row2 in ebi_df.iterrows():
                m2 = ebi_df.iloc[j]["mapping_id"]
                e2 = ebi_df.iloc[j]["full_id"]
                q2 = ebi_df.iloc[j]["query"]
                pair = sorted([m1, m2])
                if e1 != e2:
                    res = similarity = efo_nx.similarity(e1, e2).results()
                    nx_val = res[nxontology_measure]
                    data.append(
                        {
                            "m1": m1,
                            "m2": m2,
                            "e1": e1,
                            "e2": e2,
                            "q1": q1,
                            "q2": q2,
                            "nx": nx_val,
                        }
                    )
            counter += 1
        logger.info(counter)
        df = pd.DataFrame(data)
        df.to_csv(f, sep="\t", index=False)


def create_aaa():
    # run all against all for EBI query data
    m = modelData[0]
    for m in modelData:
        name = m["name"]
        f1 = f"output/{name}-ebi-encode.npy"
        f2 = f"{output}/{name}-ebi-aaa.npy"
        if os.path.exists(f2):
            logger.info(f"{name} done")
        else:
            if os.path.exists(f1):
                logger.info(m)
                dd = np.load(f1)
                logger.info(len(dd))
                aaa = create_aaa_distances(dd)
                np.save(f2, aaa)
            else:
                logger.info(f'{f1} does not exist')


def write_to_file(model_name, pairwise_data, ebi_df_all, ebi_df_filt):
    logger.info(f"writing {model_name}")
    d = []
    f = f"{output}/{model_name}-ebi-query-pairwise.tsv.gz"
    if os.path.exists(f):
        logger.info(f"Already done {f}")
    else:
        dedup_id_list = list(ebi_df_filt["mapping_id"])
        dedup_query_list = list(ebi_df_filt["query"])
        ebi_list = list(ebi_df_all["mapping_id"])
        for i in range(0, len(ebi_list)):
            if i % 100 == 0:
                logger.info(i)
            # write to file
            for j in range(i, len(ebi_list)):
                if i != j:
                    if ebi_list[i] in dedup_id_list and ebi_list[j] in dedup_id_list:
                        # if i != j:
                        score = 1 - pairwise_data[i][j]

                        # get matching query names
                        query1 = dedup_query_list[dedup_id_list.index(ebi_list[i])]
                        query2 = dedup_query_list[dedup_id_list.index(ebi_list[j])]
                        d.append(
                            {
                                "q1": query1,
                                "q2": query2,
                                "m1": ebi_list[i],
                                "m2": ebi_list[j],
                                "score": score,
                            }
                        )
        df = pd.DataFrame(d)
        logger.info(df.shape)
        df.drop_duplicates(subset=["q1", "q2"], inplace=True)
        logger.info(df.shape)
        df.to_csv(f, sep="\t", compression="gzip", index=False)


def create_pairwise(ebi_all, ebi_filt):
    # create pairwise files
    for m in modelData:
        name = m["name"]
        f = f"{output}/{name}-ebi-aaa.npy"
        if os.path.exists(f):
            dd = np.load(f"{output}/{name}-ebi-aaa.npy")
            # a=np.load('output/BioSentVec-ebi-aaa.npy')
            logger.info(len(dd))
            write_to_file(
                model_name=name,
                pairwise_data=dd,
                ebi_df_all=ebi_all,
                ebi_df_filt=ebi_filt,
            )


def create_pairwise_levenshtein(ebi_df_filt):
    f = f"{output}/Levenshtein-ebi-query-pairwise.tsv.gz"
    if os.path.exists(f):
        logger.info(f"{f} done")
    else:
        d = []
        ebi_df_filt_dic = ebi_df_filt.to_dict("records")
        for i in range(0, len(ebi_df_filt_dic)):
            for j in range(i, len(ebi_df_filt_dic)):
                if i != j:
                    distance = Levenshtein.ratio(
                        ebi_df_filt_dic[i]["query"], ebi_df_filt_dic[j]["query"]
                    )
                    d.append(
                        {
                            "q1": ebi_df_filt_dic[i]["query"],
                            "q2": ebi_df_filt_dic[j]["query"],
                            "m1": ebi_df_filt_dic[i]["mapping_id"],
                            "m2": ebi_df_filt_dic[j]["mapping_id"],
                            "score": distance,
                        }
                    )
        df = pd.DataFrame(d)
        logger.info(df.shape)
        df.drop_duplicates(subset=["q1", "q2"], inplace=True)
        logger.info(df.shape)
        df.to_csv(f, sep="\t", compression="gzip", index=False)


def create_pairwise_bert_efo(ebi_df):
    # format BERT EFO data
    f = f"{output}/BLUEBERT-EFO-ebi-query-pairwise.tsv.gz"
    if os.path.exists(f):
        logger.info(f"{f} done")
    else:
        be_df = pd.read_csv(f"data/BLUEBERT-EFO-ebi-query-pairwise.csv.gz")
        be_df['text_1'] = be_df['text_1'].str.lower()
        be_df['text_2'] = be_df['text_2'].str.lower()
        be_df.rename(columns={"text_1": "q1", "text_2": "q2"}, inplace=True)
        dedup_query_list = list(ebi_df["query"])
        be_df = be_df[
            be_df["q1"].isin(dedup_query_list) & be_df["q2"].isin(dedup_query_list)
        ]
        be_df.drop_duplicates(subset=["q1", "q2"], inplace=True)
        logger.info(be_df.shape)

        nx_df = pd.read_csv(f"{output}/nx-ebi-pairs-nr.tsv.gz", sep="\t")

        logger.info(nx_df.shape)
        logger.info(be_df.head())
        m = pd.merge(
            nx_df, be_df, left_on=["q1", "q2"], right_on=["q1", "q2"], how="left"
        )
        # need to mage values negative to use for spearman analysis against 0-1 scores
        m["score"] = m["score"] * -1
        logger.info(m.head())
        logger.info(m.shape)
        m.to_csv(f, compression="gzip", index=False, sep="\t")


def com_scores():
    # create df of scores
    # com_scores = pd.read_csv(f'{output}/nx-ebi-pairs-nr.tsv.gz',sep='\t')
    com_scores_df = pd.read_csv(f"{output}/nx-ebi-pairs-nr.tsv.gz", sep="\t")
    com_scores_df.rename(columns={"score": "nx"}, inplace=True)
    logger.info(com_scores_df.shape)
    logger.info(f"\n{com_scores_df.head()}")
    # add the distances
    for m in modelData:
        name = m["name"]
        f = f"{output}/{name}-ebi-query-pairwise.tsv.gz"
        if os.path.exists(f):
            logger.info(name)
            df = pd.read_csv(f, sep="\t")
            logger.info(f"\n{df.head()}")
            logger.info(df.shape)
            com_scores_df = pd.merge(
                com_scores_df,
                df[["m1", "m2", "score"]],
                left_on=["m1", "m2"],
                right_on=["m1", "m2"],
            )
            com_scores_df.rename(columns={"score": name}, inplace=True)
            logger.info(f"\n{com_scores_df.head()}")
    logger.info(com_scores_df.shape)
    logger.info(com_scores_df.head())
    logger.info(com_scores_df.describe())

    # drop pairs that have a missing score?
    com_scores_df.dropna(inplace=True)
    # or replace with 0?
    # com_scores.fillna(0,inplace=True)
    logger.info(com_scores_df.describe())
    logger.info(com_scores_df.shape)
    com_scores_df.to_csv(f"{output}/com_scores.tsv.gz", sep="\t", index=False)

    com_scores_df.drop(["m1", "m2", "e1", "e2", "q1", "q2"], axis=1, inplace=True)
    pearson = com_scores_df.corr(method="pearson")
    spearman = com_scores_df.corr(method="spearman")
    logger.info(f"\n{pearson}")
    ax = sns.clustermap(spearman)
    ax.savefig(f"{output}/images/spearman.png", dpi=1000)


def compare_models_with_sample(sample, term):
    logger.info(f"Comparing models with {sample}")
    # get pairs and analyse
    models = [x["name"] for x in modelData]
    models.remove("Zooma")
    models.append("nx")
    com_scores_df = pd.read_csv(f"{output}/com_scores.tsv.gz", sep="\t")
    for model in models:
        logger.info(f"Running {model}")
        com_sample = com_scores_df[
            com_scores_df["q1"].isin(sample) & com_scores_df["q2"].isin(sample)
        ][["q1", "q2", model]]
        logger.info(f'\n{com_sample}')
        # add matching pair with score 1
        for i in sample:
            df = pd.DataFrame([[i, i, 1]], columns=["q1", "q2", model])
            com_sample = com_sample.append(df)
            # logger.info(com_sample.shape)
        # add bottom half of triangle
        for i, rows in com_sample.iterrows():
            df = pd.DataFrame(
                [[rows["q2"], rows["q1"], rows[model]]], columns=["q1", "q2", model]
            )
            com_sample = com_sample.append(df)
            # logger.info(com_sample.shape)
        com_sample.drop_duplicates(inplace=True)
        #logger.info(f"\n{com_sample}")
        com_sample = com_sample.pivot(index="q1", columns="q2", values=model)
        # check for missing data
        missing = com_sample[com_sample.isna().any(axis=1)]
        logger.info(missing.shape)
        logger.info(missing.columns)
        logger.info(f'\n{missing}')
     
        com_sample = com_sample.fillna(1)

        # 1 minus to work with mantel
        n = 1 - com_sample.to_numpy()
        np.save(f"{output}/{model}-{term}.npy", n)

        logger.info(f"\n{com_sample}")
        plt.figure(figsize=(16, 7))
        sns.clustermap(com_sample, cmap="coolwarm")
        plt.savefig(f"{output}/images/sample-clustermap-{model}-{term}.png", dpi=1000)
        plt.close()


def create_random_queries():
    ran_num = 25
    com_scores_df = pd.read_csv(f"{output}/com_scores.tsv.gz", sep="\t")
    logger.info(com_scores_df.head())
    sample = list(com_scores_df["q1"].sample(n=ran_num, random_state=3))
    logger.info(sample)
    return sample


def term_sample(term):
    df = pd.read_csv(f"{output}/ebi_exact.tsv.gz", sep="\t")
    logger.info(f"\n{df.head()}")
    df = df[df["query"].str.contains(term, flags=re.IGNORECASE, regex=True)]
    logger.info(df.shape)
    logger.info(f"\n{df.head()}")

    sample = list(df["query"])[:30]
    return sample


def manual_samples():
    sample = [
        "Alzheimer s disease",
        "Abnormalities of heart beat",
        "Acute hepatitis A",
        "Sleep disorders",
        "Unspecified dementia",
        "Chronic renal failure",
        "Atopic dermatitis",
        "Crohn s disease",
        "Parkinson s disease",
        "Cushing s syndrome",
        "Secondary parkinsonism",
        "Huntington s disease",
        "Pulmonary valve disorders",
        "Snoring",
        "Pulse rate",
        "Body mass index (BMI)",
        "Whole body fat mass",
        "heart/cardiac problem",
        "sleep apnoea",
        "high cholesterol",
        "heart valve problem/heart murmur",
        "other renal/kidney problem",
        "colitis/not crohns or ulcerative colitis",
    ]
    sample = [x.lower() for x in sample]
    return sample


def run_mantel(term):
    models = [x["name"] for x in modelData]
    models.remove("Zooma")
    models.append("nx")
    d = []
    for i in range(0, len(models)):
        for j in range(0, len(models)):
            # if i != j:
            m1 = models[i]
            m2 = models[j]
            logger.info(f"{m1} {m2}")
            n1 = np.load(f"{output}/{m1}-{term}.npy")
            # logger.info(n1.shape)
            n2 = np.load(f"{output}/{m2}-{term}.npy")
            # logger.info(n2.shape)
            coeff, p_value, n = mantel(x=n1, y=n2, method="pearson")
            logger.info(f"{coeff} {p_value} {n}")
            d.append({"m1": m1, "m2": m2, "coeff": coeff, "p_value": p_value})
    df = pd.DataFrame(d)
    logger.info(f"\n{df}")
    df = df.pivot(index="m1", columns="m2", values="coeff")
    logger.info(f"\n{df}")
    plt.figure(figsize=(16, 7))
    sns.clustermap(df, cmap="coolwarm")
    plt.savefig(f"{output}/images/mantel-{term}.png", dpi=1000)
    plt.close()


def sample_checks():

    term = "neoplasm"
    sample = term_sample(term=term)
    compare_models_with_sample(sample=sample, term=term)
    run_mantel(term)

    term = "random"
    sample = create_random_queries()
    compare_models_with_sample(sample=sample, term=term)
    run_mantel(term)

    term = "manual"
    sample = manual_samples()
    compare_models_with_sample(sample=sample, term="manual")
    run_mantel(term)


def run_all():
    efo_nx = create_nx()
    ebi_all, ebi_filt = read_ebi()
    # create_nx_pairs_nr(ebi_df,efo_nx)
    create_nx_pairs_nr(ebi_filt, efo_nx)
    create_aaa()
    create_pairwise(ebi_all, ebi_filt)
    create_pairwise_bert_efo(ebi_filt)
    create_pairwise_levenshtein(ebi_filt)
    com_scores()
    sample_checks()


def dev():
    # efo_nx = create_nx()
    ebi_all,ebi_filt = read_ebi()
    # create_nx_pairs_nr(ebi_filt,efo_nx)
    # create_aaa()
    # create_pairwise(ebi_all,ebi_filt)
    # create_pairwise_bert_efo(ebi_filt)
    # create_pairwise_levenshtein(ebi_filt)
    #com_scores()
    #sample_checks()


if __name__ == "__main__":
    #dev()
    run_all()
