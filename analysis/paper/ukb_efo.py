#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import json
import time
import os
import gzip
import timeit
import scispacy
import spacy
import Levenshtein
from scripts.vectology_functions import (
    create_aaa_distances,
    create_pair_distances,
    embed_text,
    create_efo_nxo,
    create_efo_data,
)
from loguru import logger
from pathlib import Path

import seaborn as sns

# Apply the default theme
sns.set_theme()

# globals
ebi_data = "data/UK_Biobank_master_file.tsv"
efo_nodes_v1 = "data/efo_nodes_2021_02_01.csv"
efo_rels_v1 = "data/efo_edges_2021_02_01.csv"
efo_nodes_v2 = "data/efo_nodes_2021_05_26.csv"
efo_rels_v2 = "data/efo_edges_2021_05_26.csv"
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

# set up an output directory
output = "output/trait-efo-v1-lowercase"
Path(output).mkdir(parents=True, exist_ok=True)
Path(f"{output}/images").mkdir(parents=True, exist_ok=True)

# get the UKB EFO mappings and EFO data
def get_data():
    # get the EBI UKB data
    if not os.path.exists(ebi_data):
        logger.info(f"Downloading {ebi_data}...")
        os.system(
            "wget -O data/UK_Biobank_master_file.tsv https://raw.githubusercontent.com/EBISPOT/EFO-UKB-mappings/master/UK_Biobank_master_file.tsv"
        )
    # get the EFO data
    if not os.path.exists(efo_nodes):
        logger.info(f"Downloading efo.json...")
        os.system(
            "wget -O data/efo.json https://github.com/EBISPOT/efo/releases/download/v3.29.1/efo.json"
        )
        node_df, edge_df = create_efo_data("data/efo.json")
        node_df.to_csv(efo_nodes, index=False)
        edge_df.to_csv(efo_rels, index=False)


# read EFO node data
def efo_node_data_v2():
    # get EFO node data
    df = pd.read_csv(efo_nodes)
    df.rename(columns={"lbl": "efo_label", "id": "efo_id"}, inplace=True)
    # drop type
    df.drop(["definition", "umls"], inplace=True, axis=1)
    # lowercase the label
    df['efo_label'] = df['efo_label'].str.lower()
    df.drop_duplicates(inplace=True)
    logger.info(f"\n{df}")
    logger.info({df.shape})
    return df


def efo_node_data_v1():
    # get EFO node data
    df = pd.read_csv(efo_nodes_v1)
    df.rename(columns={"efo.value": "efo_label", "efo.id": "efo_id"}, inplace=True)
    # drop type
    df.drop(["efo.type"], inplace=True, axis=1)
    # lowercase the label
    df['efo_label'] = df['efo_label'].str.lower()
    # drop duplicates by name 
    df.drop_duplicates(subset=['efo_label'],inplace=True)
    logger.info(f"\n{df}")
    logger.info({df.shape})
    return df


# read the EBI mapping data
def get_ebi_data(efo_node_df):
    f = "output/ebi-ukb-cleaned-cat.csv"
    if os.path.exists(f):
        logger.info(f"{f} exists")
        ebi_df = pd.read_csv(f)
        logger.info(ebi_df.shape)
        logger.info(f'\n{ebi_df["MAPPING_TYPE"].value_counts()}')
    else:
        ebi_df = pd.read_csv(ebi_data, sep="\t")

        # drop some columns
        ebi_df = ebi_df[
            ["ZOOMA QUERY", "MAPPED_TERM_LABEL", "MAPPED_TERM_URI", "MAPPING_TYPE"]
        ]
        ebi_df.rename(columns={"ZOOMA QUERY": "query"}, inplace=True)
        logger.info(f"\n{ebi_df.head()}")
        logger.info(ebi_df.shape)

        # drop rows with multiple mappings
        ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains(",", na=False)]
        logger.info(ebi_df.shape)
        ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains("\|", na=False)]
        logger.info(ebi_df.shape)

        # clean up and lowercase
        ebi_df["id"] = ebi_df["MAPPED_TERM_URI"].str.strip()
        ebi_df["query"] = ebi_df["query"].str.lower()

        # remove underscores
        ebi_df["query"] = ebi_df["query"].str.replace("_", " ")

        # drop where query and id are duplicates
        ebi_df.drop_duplicates(subset=["query", "id"], inplace=True)
        logger.info(ebi_df.shape)

        # drop nan
        ebi_df.dropna(inplace=True)
        logger.info(ebi_df.shape)
        logger.info(ebi_df.head())

        # drop cases where query and matched text are identical
        logger.info(ebi_df.shape)
        ebi_df = ebi_df[
            ebi_df["query"].str.lower() != ebi_df["MAPPED_TERM_LABEL"].str.lower()
        ]
        logger.info(ebi_df.shape)

        # get counts of mapping type
        logger.info(f'\n{ebi_df["MAPPING_TYPE"].value_counts()}')

        # check data against efo nodes
        efo_node_ids = list(efo_node_df["efo_id"])
        ebi_ids = list(ebi_df["id"])
        missing = []
        matched = []
        for i in ebi_ids:
            match = False
            for s in efo_node_ids:
                if i in s and match == False:
                    matched.append(s)
                    match = True
            if match == False:
                missing.append(i)
        logger.info(f"Missing: {len(missing)} {missing}")

        # remove missing from ukb data
        logger.info(f"\n{ebi_df.head()}")
        logger.info(ebi_df.shape)
        for i in missing:
            ebi_df = ebi_df.drop(ebi_df[ebi_df["id"].str.contains(i)].index)
        ebi_df["full_id"] = matched

        # add index as ID
        ebi_df["mapping_id"] = range(1, ebi_df.shape[0] + 1)
        logger.info(ebi_df.head())
        logger.info(ebi_df.shape)
        ebi_df.to_csv(f, index=False)
    logger.info(f"\n{ebi_df.head()}")
    return ebi_df


# encode the EBI query terms with Vectology models
def encode_ebi(ebi_df):
    queries = list(ebi_df["query"])
    chunk = 10

    vectology_models = ["BioSentVec", "BioBERT", "BlueBERT"]

    for m in modelData:
        name = m["name"]
        model = m["model"]
        if name in vectology_models:
            f = f"output/{m['name']}-ebi-encode.npy"
            if os.path.exists(f):
                logger.info(f"{name} done")
            else:
                logger.info(f"Encoding EBI queriues with {model}")
                results = []
                for i in range(0, len(queries), chunk):
                    if i % 100 == 0:
                        logger.info(i)
                    batch = queries[i : i + chunk]
                    res = embed_text(textList=batch, model=model)
                    for r in res:
                        results.append(r)
                logger.info(f"Results {len(results)}")
                np.save(f, results)


# embed the efo node names with Vectology models
def encode_efo(efo_node_df):
    queries = list(efo_node_df["efo_label"])
    chunk = 20

    vectology_models = ["BioSentVec", "BioBERT", "BlueBERT"]

    for m in modelData:
        name = m["name"]
        model = m["model"]
        if name in vectology_models:
            f = f"output/{m['name']}-efo-encode.npy"
            if os.path.exists(f):
                logger.info(f"{name} done")
            else:
                logger.info(f"Encoding EFO queriues with {model}")
                results = []
                for i in range(0, len(queries), chunk):
                    if i % 100 == 0:
                        logger.info(i)
                    batch = queries[i : i + chunk]
                    res = embed_text(textList=batch, model=model)
                    for r in res:
                        results.append(r)
                # logger.info(f'Results {results}')
                np.save(f, results)


# create GUSE embeddings for EBI and EFO
def run_guse(ebi_df, efo_node_df):
    f1 = "output/GUSE-ebi-encode.npy"
    f2 = "output/GUSE-efo-encode.npy"

    if os.path.exists(f2):
        logger.info(f"{f2} done")
    else:
        # Google Universal Sentence Encoder

        #!pip install  "tensorflow>=2.0.0"
        #!pip install  --upgrade tensorflow-hub

        import tensorflow_hub as hub

        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        guse_ebi_embeddings = embed(ebi_df["query"])
        guse_efo_embeddings = embed(efo_node_df["efo_label"])

        guse_ebi_embeddings_list = []
        for g in guse_ebi_embeddings:
            guse_ebi_embeddings_list.append(g.numpy())
        np.save(f1, guse_ebi_embeddings_list)
        # ebi_df['GUSE']=guse_ebi_embeddings_list

        guse_efo_embeddings_list = []
        for g in guse_efo_embeddings:
            guse_efo_embeddings_list.append(g.numpy())
        np.save(f2, guse_efo_embeddings_list)
        logger.info("GUSE done")


# create SPACY and SciSpaCy embeddings
def run_spacy(model, name, ebi_df, efo_node_df):
    f1 = f"output/{name}-ebi-encode.npy"
    f2 = f"output/{name}-efo-encode.npy"
    if os.path.exists(f2):
        logger.info(f"{f2} exists")
    else:
        logger.info(f"loading {model}")
        nlp = spacy.load(model)
        ebi_query_docs = list(nlp.pipe(ebi_df["query"]))
        efo_label_docs = list(nlp.pipe(efo_node_df["efo_label"]))

        spacy_ebi_embeddings_list = []
        for g in ebi_query_docs:
            spacy_ebi_embeddings_list.append(g.vector)
        np.save(f1, spacy_ebi_embeddings_list)

        spacy_efo_embeddings_list = []
        for g in efo_label_docs:
            spacy_efo_embeddings_list.append(g.vector)
        np.save(f2, spacy_efo_embeddings_list)


def run_levenshtein(ebi_df, efo_node_df):
    f = f"{output}/levenshtein-pairwise.tsv.gz"
    if os.path.exists(f):
        logger.info(f"{f} done")
    else:
        d = []
        ebi_dic = ebi_df.to_dict("records")
        efo_dic = efo_node_df.to_dict("records")
        for i in range(0, len(ebi_dic)):
            if i % 100 == 0:
                logger.info(i)
            for j in range(0, len(efo_dic)):
                query = ebi_dic[i]["query"]
                efo_label = efo_dic[j]["efo_label"]
                distance = Levenshtein.ratio(query, efo_label)
                d.append(
                    {
                        "mapping_id": ebi_dic[i]["mapping_id"],
                        "manual": ebi_dic[i]["full_id"],
                        "prediction": efo_dic[j]["efo_id"],
                        "score": distance,
                    }
                )
        df = pd.DataFrame(d)
        logger.info(df.head())
        df.to_csv(f, sep="\t", index=False, compression="gzip")


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


# create cosine pairs
def run_pairs(model):
    dd_name = f"{output}/{model}-dd.npy"

    # v1 = list(ebi_df[model])
    # v2 = list(efo_df[model])
    if os.path.exists(f"output/{model}-ebi-encode.npy"):
        v1 = np.load(f"output/{model}-ebi-encode.npy")
        v2 = np.load(f"output/{model}-efo-encode.npy")

        if os.path.exists(dd_name):
            logger.info(f"{dd_name} already created, loading...")
            with open(dd_name, "rb") as f:
                dd = np.load(f)
        else:
            # cosine of lists
            dd = create_pair_distances(v1, v2)
            np.save(dd_name, dd)
        logger.info("done")
        return dd


# parse cosine paris and write to file
def write_to_file(model_name, pairwise_data, ebi_df, efo_node_df):
    logger.info(f"writing {model_name}")
    f = f"{output}/{model_name}-pairwise.tsv.gz"
    if os.path.exists(f):
        logger.info(f"Already done {f}")
    else:
        ebi_efo_list = list(ebi_df["full_id"])
        efo_list = list(efo_node_df["efo_id"])
        d = []
        for i in range(0, len(ebi_efo_list)):
            if i % 100 == 0:
                logger.info(i)
            # write to file
            mCount = 0
            for j in range(i, len(efo_list)):
                # if i != j:
                score = 1 - pairwise_data[i][j]
                d.append(
                    {
                    'mapping_id':i+1,
                    'manual':ebi_efo_list[i],
                    'prediction':efo_list[j],
                    'score':score
                    }
                )
                mCount += 1
        df = pd.DataFrame(d)
        df.to_csv(f,sep='\t',index=False)


# wrapper for above
def create_pair_data(ebi_df, efo_node_df):
    for m in modelData:
        logger.info(m["name"])
        dd = run_pairs(m["name"])
        if dd is not None:
            # get_top(m['name'],dd)
            write_to_file(
                model_name=m["name"],
                pairwise_data=dd,
                ebi_df=ebi_df,
                efo_node_df=efo_node_df,
            )
        else:
            logger.info(f'{m["name"]} not done')


# zooma API
def zooma_api(text):
    zooma_api_url = "https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate"
    payload = {"propertyValue": text, "filter": "required:[none],ontologies:[efo]"}
    res = requests.get(zooma_api_url, params=payload).json()
    if res:
        # logger.info(res)
        efo = res[0]["semanticTags"][0]
        confidence = res[0]["confidence"]
        return {
            "query": text,
            "prediction": efo.strip(),
            "confidence": confidence.strip(),
        }
    else:
        return {"query": text, "prediction": "NA", "confidence": "NA"}


# run zooma API for ebi mappings
def run_zooma(ebi_df):
    # takes around 3 minutes for 1,000
    f = f"{output}/zooma.tsv"
    if os.path.exists(f):
        logger.info("Zooma done")
    else:
        all_res = []
        count = 1
        for q in list(ebi_df["query"]):
            res = zooma_api(q)
            res["mapping_id"] = count
            all_res.append(res)
            count += 1
        df = pd.DataFrame(all_res)
        # df['manual'] = ebi_df['full_id'][0:5]
        # df['mapping_id'] = ebi_df['mapping_id'][0:5]
        logger.info(df.head())
        # df = pd.merge(df,ebi_df,left_on='query',right_on='query')
        logger.info(df.shape)
        # df[['mapping_id','manual','prediction','confidence']].to_csv(f,index=False,sep='\t')
        df.to_csv(f, index=False, sep="\t")


# filter the zooma output
def filter_zooma(efo_nx, ebi_df):
    dis_results = []
    efo_results = []
    df = pd.read_csv(f"{output}/zooma.tsv", sep="\t")
    # add manual EFO
    df = pd.merge(
        df,
        ebi_df[["mapping_id", "full_id"]],
        left_on="mapping_id",
        right_on="mapping_id",
    )
    df.rename(columns={"full_id": "manual"}, inplace=True)
    logger.info(df.head())
    for i, row in df.iterrows():
        try:
            res = similarity = efo_nx.similarity(
                row["prediction"], row["manual"]
            ).results()
            dis_results.append(res[nxontology_measure])
        except:
            dis_results.append(0)

    logger.info(len(dis_results))
    df["score"] = dis_results
    logger.info(df[df["score"] > 0.9].shape)
    df.to_csv(
        f"{output}/Zooma-pairwise-filter.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    # sns.displot(df, x="score",kde=True)


# create filtered pairwise data
def filter_paiwise_file(model_name):
    logger.info(f"filter_pairwise_file {model_name}")
    f = f"{output}/{model_name}-pairwise-filter.tsv.gz"
    if os.path.exists(f):
        logger.info(f"Already done {model_name}")
        return
    else:
        try:
            df = pd.read_csv(f"{output}/{model_name}-pairwise.tsv.gz", sep="\t")
            logger.info(df.shape)
            df = (
                df.sort_values(by=["score"], ascending=False)
                .groupby("mapping_id")
                .head(top_x)
            )
            # drop duplicates
            df.drop_duplicates(
                subset=["mapping_id", "manual", "prediction"], inplace=True
            )
            logger.info(df.shape)
            df.sort_values(by=["mapping_id", "score"], ascending=[True, False]).to_csv(
                f, sep="\t", index=False, compression="gzip"
            )
        except:
            logger.info(f"Error {model_name}")
            return


# read BLUEBERT-EFO data and filter
def filter_bert(ebi_df, efo_node_df):
    df = pd.read_csv(f"data/efo_mk1_inference_top100_no_underscores.csv.gz")
    # lowercase
    df['text_1'] = df['text_1'].str.lower()
    df['text_2'] = df['text_2'].str.lower()
    df_top = df.sort_values(by=["score"]).groupby("text_1").head(top_x)
    df_top = pd.merge(
        df_top,
        ebi_df[["mapping_id", "query", "full_id"]],
        left_on="text_1",
        right_on="query",
    )
    df_top.rename(columns={"full_id": "manual"}, inplace=True)
    df_top.drop("query", axis=1, inplace=True)

    # map to predicted EFO
    df_top = pd.merge(df_top, efo_node_df, left_on="text_2", right_on="efo_label")
    df_top.rename(columns={"efo_id": "prediction"}, inplace=True)
    df_top.drop("efo_label", axis=1, inplace=True)
    df_top.drop_duplicates(subset=["mapping_id", "manual", "prediction"], inplace=True)
    logger.info(df_top.head())
    df_top[["mapping_id", "manual", "prediction", "score"]].sort_values(
        by=["mapping_id", "score"]
    ).to_csv(
        f"{output}/BLUEBERT-EFO-pairwise-filter.tsv.gz",
        index=False,
        compression="gzip",
        sep="\t",
    )
    logger.info(df_top.head())


# create top pairs
def get_top_using_pairwise_file(model_name, top_num, efo_nx, ebi_df):
    f = f"{output}/{model_name}-top-{top_num}.tsv.gz"
    if os.path.exists(f):
        logger.info(f"Top done {model_name}")
    else:
        logger.info(f"Reading {model_name}")
        try:
            df = pd.read_csv(f"{output}/{model_name}-pairwise-filter.tsv.gz", sep="\t")
        except:
            logger.info("Data do not exist for", model_name)
            return
        logger.info(df.head())
        logger.info(df.shape)
        # remove duplicates
        #df.drop_duplicates(subset=["mapping_id","manual", "prediction"], inplace=True)
        top_res = []
        #mapping_ids = list(ebi_df["mapping_id"].unique())
        for i,row_i in ebi_df.iterrows():
            # for i in range(0,10):
            mapping_id = row_i['mapping_id']
            efo_predictions = df[df["mapping_id"] == mapping_id].head(n=top_num)[
                ["prediction", "score"]
            ]
            # end = time.time()
            # run nxontology for each
            for j, row_j in efo_predictions.iterrows():
                manual_efo = row_i['full_id']
                predicted_efo = row_j["prediction"]
                score = row_j["score"]
                try:                        
                    res = efo_nx.similarity(manual_efo, predicted_efo).results()
                    nx_val = res[nxontology_measure]
                except:
                    nx_val = 0
                top_res.append(
                    {
                        "mapping_id": i + 1,
                        "manual": row_i['full_id'],
                        "prediction": predicted_efo,
                        "score": score,
                        "nx": nx_val,
                    }
                )
        res_df = pd.DataFrame(top_res)
        res_df.to_csv(f, index=False, sep="\t", compression="gzip")


# calculate weighted average
def calc_weighted_average(model_name, top_num, mapping_types, ebi_df):
    f = f"{output}/{model_name}-top-100.tsv.gz"
    logger.info(f)
    res = []
    try:
        df = pd.read_csv(f, sep="\t")
    except:
        logger.info(f"Data do not exist for {model_name}")
        return
    manual_efos = list(ebi_df["full_id"])
    for i in range(0, len(manual_efos)):
        manual_efo = manual_efos[i]
        # filter on type
        mapping_type = ebi_df[ebi_df["full_id"] == manual_efo]["MAPPING_TYPE"].values[0]
        if mapping_type in mapping_types:
            nx_scores = list(df[df["manual"] == manual_efo].head(n=top_num)["nx"])
            weights = list(reversed(range(1, (len(nx_scores) + 1))))
            try:
                weighted_avg = round(np.average(nx_scores, weights=weights), 3)
            except:
                weighted_avg = 0
            res.append(weighted_avg)
    logger.info(len(res))
    return res


# run a range of weighted averages
def run_wa(mapping_types, mapping_name, ebi_df):
    top_nums = [1, 2, 5, 10, 20, 50, 100]
    for top_num in top_nums:
        f = f"{output}/images/weighted-average-nx-{top_num}-{mapping_name}.png"
        if os.path.exists(f):
            logger.info(f"{f} done")
        else:
            all_res = {}
            for m in modelData:
                if top_num > 1 and m["name"] == "Zooma":
                    logger.info("No data for Zooma")
                    continue
                else:
                    res = calc_weighted_average(
                        m["name"], top_num, mapping_types, ebi_df
                    )
                    if res is not None:
                        all_res[m["name"]] = res

            df = pd.DataFrame(all_res)
            df["efo"] = ebi_df["full_id"]
            logger.info(f'\n{df.head()}')
            df_melt = pd.melt(df, id_vars=["efo"])
            df_melt.rename(columns={"variable": "Model"}, inplace=True)
            df_melt.to_csv(f'{output}/weighted-average-nx-{top_num}-{mapping_name}.csv',index=False)
            ax = sns.displot(
                x="value",
                hue="Model",
                data=df_melt,
                kind="kde",
                cut=0,
                palette=palette,
                height=6,
                common_norm=True,
            )
            # sns_plot = sns.displot(ebi_df, x=f"{model}-nx",kde=True)
            ax.set(xlabel=f"Weighted average of nx", ylabel="Density")
            # ax.set_xscale("log")
            ax.savefig(f, dpi=1000)

def describe_wa(wa_csv):
    df = pd.read_csv(wa_csv)
    logger.info(f'\n{df.head()}') 
    d = df.groupby('Model').describe()
    logger.info(d)
    # violin plot
    mean_order = df.groupby('Model')['value'].mean().reset_index().sort_values('value',ascending=False)['Model']
    logger.info(mean_order)
    ax = sns.catplot(x="Model",y="value",
               data=df, kind="violin", order=mean_order, palette=palette)
    ax.set(xlabel=f"Model/Method",ylabel="Weighted average of top 10 batet scores")
    ax.set_xticklabels(rotation=45, ha="right")
    # ax.set_xscale("log")
    ax.savefig(f'{output}/images/wa-violin-plot.png', dpi=1000)


# create plot of number of correct top predictions for each model
def get_top_hits(ebi_df, batet_score=1,category=''):
    fig_f = f"{output}/images/top-counts-batet-{batet_score}-{category}.png"
    if os.path.exists(fig_f):
        logger.info(f'{fig_f} exists')
        return
    logger.info(f'{category} {batet_score}')
    logger.info(ebi_df.shape)

    # stratify by category
    if category != 'all':
        ebi_df = ebi_df[ebi_df['Type']==category]
        logger.info(ebi_df.shape)

    res = []
    # add manual mapping info
    ebi_df.loc[~ebi_df["MAPPING_TYPE"].isin(["Exact", "Broad", "Narrow"]), "MAPPING_TYPE"] = "Other"
    d = dict(ebi_df["MAPPING_TYPE"].value_counts())
    d["Model"] = "Manual"
    d["Total"] = ebi_df.shape[0]
    res.append(d)

    # for each model get check top hits
    for i in modelData:
        fName = f"{output}/{i['name']}-top-100.tsv.gz"
        logger.info(fName)
        df = pd.read_csv(fName, sep="\t")
        df = pd.merge(
            df,
            ebi_df[["mapping_id", "MAPPING_TYPE"]],
            left_on="mapping_id",
            right_on="mapping_id",
        )
        df.drop_duplicates(subset=["mapping_id", "manual"], inplace=True)

        # add "Other" MAPPING_TYPE
        df.loc[~df["MAPPING_TYPE"].isin(["Exact", "Broad", "Narrow"]), "MAPPING_TYPE"] = "Other"
        logger.info(f'\n{df.columns}')

        # filter by mapping_type
        d = dict(df[df["nx"] >= batet_score]["MAPPING_TYPE"].value_counts())
        d["Model"] = i["name"]
        d["Total"] = df[df["nx"] >= batet_score].shape[0]
        res.append(d)

    #logger.info(res)
    res_df = pd.DataFrame(res).sort_values(by="Total", ascending=False)
    totals = list(res_df['Total'])
    # drop totals from plot
    res_df.drop(columns=['Total'],inplace=True)
    logger.info(res_df.columns)
    logger.info(res_df)
    logger.info(res_df.shape)
    #if res_df.shape[1]==1:
    #    logger.warning('No data for plot')
    #    return
    ax = res_df.plot.bar(
        stacked=True, figsize=(10, 10)
    )
    ax.set_xticklabels(res_df["Model"], rotation=45, ha="right")
    
    # add totals
    logger.info(totals)
    # Set an offset that is used to bump the label up a bit above the bar.
    y_offset = 4
    # Add labels to each bar.
    for i, total in enumerate(totals):
        logger.info(f'{i} {total}')
        ax.text(i, total + y_offset, round(total), ha='center',weight='bold')
    
    fig = ax.get_figure()
    fig.savefig(fig_f, dpi=1000)


# look at examples where predictions vary across model
#  high = all high nx
# low = all low nx
# spread = high sd nx
def run_high_low(type: str, ebi_df_exact, efo_node_df):
    f = f"{output}/{type}-predictions.tsv"
    if os.path.exists(f):
        logger.info(f'{f} done')
        return
    match = []
    logger.info(f"\n##### {type} #####")
    for i in modelData:
        fName = f"{output}/{i['name']}-top-100.tsv.gz"
        logger.info(fName)
        df = pd.read_csv(fName, sep="\t")
        # maybe filter on exact mapping type
        # df = df[df['mapping_id'].isin(exact_mapping_type)]
        df.drop_duplicates(subset=["mapping_id", "manual"], inplace=True)

        if type == "low":
            match.extend(list(set(list(df[df["nx"] > 0.95]["mapping_id"]))))
            hl_df = ebi_df_exact[~ebi_df_exact["mapping_id"].isin(match)]

        elif type == "high":
            match.extend(list(set(list(df[df["nx"] < 0.95]["mapping_id"]))))
            hl_df = ebi_df_exact[~ebi_df_exact["mapping_id"].isin(match)]

        elif type == "spread":
            hl_df = ebi_df_exact

    logger.info(hl_df.shape)
    logger.info(hl_df["MAPPING_TYPE"].value_counts())

    # add the top prediction from each model to the missing df
    for i in modelData:
        fName = f"{output}/{i['name']}-top-100.tsv.gz"
        logger.info(fName)
        df = pd.read_csv(fName, sep="\t")
        df.drop_duplicates(subset=["mapping_id", "manual"], inplace=True)
        # get data
        hl_df = pd.merge(
            df[["mapping_id", "prediction", "score", "nx"]],
            hl_df,
            left_on="mapping_id",
            right_on="mapping_id",
        )
        hl_df.rename(
            columns={
                "prediction": f"{i['name']}-efo",
                "score": f"{i['name']}-score",
                "nx": f"{i['name']}-nx",
            },
            inplace=True,
        )

    # get rows with largest range for spread option
    if type == "spread":
        nx_vals = hl_df.filter(regex=".*-nx", axis=1)
        logger.info(nx_vals)
        nx_vals["std"] = nx_vals.std(axis=1)
        logger.info(nx_vals)
        # get top 10 range
        nx_vals_top = nx_vals.sort_values(by="std", ascending=False)
        logger.info(nx_vals_top)
        nx_ind = nx_vals_top.index
        logger.info(nx_ind)
        hl_df = hl_df[hl_df.index.isin(nx_ind)]
        hl_df["std"] = nx_vals["std"]
        hl_df = hl_df.sort_values("std", ascending=False)

    # add mean nx
    hl_df["mean-nx"] = hl_df.filter(regex=".*-nx", axis=1).mean(
        numeric_only=True, axis=1
    )
    # sort by min nx
    if type == "high":
        hl_df.sort_values(by="mean-nx", ascending=False, inplace=True)
    elif type == "low":
        hl_df.sort_values(by="mean-nx", ascending=True, inplace=True)

    # add EFO labels
    efo_cols = [col for col in hl_df.columns if "efo" in col]
    efo_dic = dict(zip(efo_node_df["efo_id"], efo_node_df["efo_label"]))
    for e in efo_cols:
        model = e.replace("-efo", "")
        model_efo_name = f"{model}-efo-name"
        hl_df[model_efo_name] = hl_df[e].map(efo_dic)

    logger.info(hl_df.head())
    hl_df.to_csv(f, sep="\t", index=False)


# output tidy files from above
def tidy_up_and_get_rank(df, efo_node_df, name):
    efo_cols = [col for col in df.columns if col.endswith("efo")]
    #efo_dic = dict(zip(efo_node_df["efo_id"], efo_node_df["efo_label"]))
    # logger.info(efo_cols)
    keep_list = ["query", "MAPPED_TERM_LABEL", "Type"]
    # efo_cols=['SequenceMatcher-efo']
    for e in efo_cols:
        logger.info(e)
        model = e.replace("-efo", "")
        keep_list.append(model)
        model_efo_name = f"{model}-efo-name"
        model_nx = f"{model}-nx"
        #logger.info(f"{df[e]} {df[e].map(efo_dic)}")
        #df[model_efo_name] = df[e].map(efo_dic)
        # get the rank
        top = pd.read_csv(f"{output}/{model}-top-100.tsv.gz", sep="\t")
        #ogger.info(top)
        rank_vals = []
        for i, row in df.iterrows():
            mapping_id = row["mapping_id"]
            full_id = row["full_id"]
            #logger.info(f"#### {full_id}")
            top_match_df = top[top["mapping_id"] == mapping_id].reset_index()
            # find manual efo in top
            match_rank = top_match_df[top_match_df["prediction"] == full_id]
            if match_rank.empty:
                match_rank = ">100"
            else:
                match_rank = match_rank.index[0].item() + 1
            rank_vals.append(
                f"{row[model_efo_name]} ({match_rank}) [{round(row[model_nx],2)}]"
            )
        df[model] = rank_vals
    df = df[keep_list]
    logger.info(f"\n{df}")
    df.to_csv(f"{output}/all-{name}.csv", index=False)
    return df


# run the high/low/spread examples
def create_examples(efo_node_df):
    ebi_df = pd.read_csv("output/ebi-ukb-cleaned-cat.csv")
    ebi_df_exact = ebi_df[ebi_df["MAPPING_TYPE"] == "Exact"]
    logger.info(ebi_df_exact.shape)

    logger.info(f"\n{ebi_df_exact}")
    ebi_df_exact.drop_duplicates(subset=["query"], inplace=True)
    logger.info(ebi_df_exact.shape)

    run_high_low("low", ebi_df_exact, efo_node_df)
    run_high_low("high", ebi_df_exact, efo_node_df)
    run_high_low("spread", ebi_df_exact, efo_node_df)

    df_low = pd.read_csv(f"{output}/low-predictions.tsv", sep="\t")
    logger.info(f"\n{df_low.head()}")
    logger.info(f"\n{df_low.columns}")
    df_low = tidy_up_and_get_rank(df_low, efo_node_df, "low")

    df_high = pd.read_csv(f"{output}/high-predictions.tsv", sep="\t")
    df_high = tidy_up_and_get_rank(df_high, efo_node_df, "high")

    df_range = (
        pd.read_csv(f"{output}/spread-predictions.tsv", sep="\t")
        .sort_values("std", ascending=False)
        .head(n=20)
    )
    df_range = tidy_up_and_get_rank(df_range, efo_node_df, "spread")


def run():
    # get efo node data
    efo_node_df = efo_node_data_v1()
    # get ebi efo data
    ebi_df = get_ebi_data(efo_node_df)
    # create embeddings for ebi using vectology models
    encode_ebi(ebi_df)
    # create embedings for efo using vectology models
    encode_efo(efo_node_df)
    # run Google Universal
    run_guse(ebi_df, efo_node_df)
    # run spacy
    run_spacy(
        model="en_core_web_lg", name="Spacy", ebi_df=ebi_df, efo_node_df=efo_node_df
    )
    run_spacy(
        model="en_core_sci_lg", name="SciSpacy", ebi_df=ebi_df, efo_node_df=efo_node_df
    )
    # run sequencematcher
    run_levenshtein(ebi_df, efo_node_df)
    # create nxontology
    efo_nx = create_nx()
    # run pairwise cosine distance
    create_pair_data(ebi_df, efo_node_df)
    # run zooma
    run_zooma(ebi_df)
    filter_zooma(efo_nx, ebi_df)
    # filter BERT-EFO results
    filter_bert(ebi_df=ebi_df, efo_node_df=efo_node_df)
    # filter pairwise data
    for m in modelData:
        filter_paiwise_file(model_name=m["name"])
        get_top_using_pairwise_file(
            model_name=m["name"], top_num=100, efo_nx=efo_nx, ebi_df=ebi_df
        )
    # create summary weighted average plots
    run_wa(
        mapping_types=["Exact", "Broad", "Narrow"], mapping_name="all", ebi_df=ebi_df
    )
    run_wa(mapping_types=["Exact"], mapping_name="exact", ebi_df=ebi_df)
    run_wa(
        mapping_types=["Broad", "Narrow"], mapping_name="broad-narrow", ebi_df=ebi_df
    )
    # create summary tophits plot
    cats = set(list(list(ebi_df['Type'])))
    # run over a range of batet filters
    for i in range(5,11):
        # run for each variable category
        for c in cats:
            get_top_hits(ebi_df,batet_score=i/10,category=c)
        # run for all cats
        get_top_hits(ebi_df,batet_score=i/10,category='all')
    # create high/low/spread tables
    create_examples(efo_node_df)


def dev():
    efo_node_df = efo_node_data_v1()
    ebi_df = get_ebi_data(efo_node_df)
    describe_wa(f'{output}/weighted-average-nx-10-all.csv')
if __name__ == "__main__":
    #run()
    dev()
