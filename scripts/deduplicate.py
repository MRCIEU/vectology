import pandas as pd
import numpy as np
import os
import sknetwork as skn
from sklearn.cluster import DBSCAN

from loguru import logger
from scripts.vectology_functions import embed_text, create_aaa_distances

output = 'paper/output/deduplication'
os.makedirs(output, exist_ok=True)

file_name = 'paper/data/UK_Biobank_master_file.tsv'
trait_npy = f'{output}/trait_list.npy'
aaa_npy = f'{output}/aaa.npy'

def read_traits():
    df = pd.read_csv(file_name,sep='\t')
    df.rename(columns={"ZOOMA QUERY": "query"}, inplace=True)
    df = df.filter(['query'])
    # lowercase
    df["query"] = df["query"].str.lower()
    # remove underscores
    df["query"] = df["query"].str.replace("_", " ")
    # drop duplicates
    df.drop_duplicates(inplace=True)
    logger.info(f'\n{df}')
    logger.info(df.shape)
    return df

def encode_gwas(df):
    if os.path.exists(trait_npy):
        logger.info(f'{trait_npy} done')
    else:
        queries = list(df["query"])
        chunk = 10
        results = []
        for i in range(0, len(queries), chunk):
            if i % 100 == 0:
                logger.info(i)
            batch = queries[i : i + chunk]
            res = embed_text(textList=batch, model='BioSentVec')
            for r in res:
                results.append(r)
        logger.info(f"Results {len(results)}")
        np.save(trait_npy, results)

def run_aaa(df):
    if os.path.exists(aaa_npy):
        logger.info(f'{aaa_npy} exists')
    else:
        dd = np.load(trait_npy)
        aaa = create_aaa_distances(dd)
        # set nan to 0
        where_are_NaNs = aaa[np.isnan(aaa)]=0
        # check for nan
        logger.info(np.any(np.isnan(aaa)))
        logger.info(aaa.shape)
        np.save(aaa_npy,aaa)

def cluster(df,eps,cluster_file):
    aaa = np.load(aaa_npy)
    if os.path.exists(cluster_file):
        logger.info(f'{cluster_file} exists')
    else:
        logger.info(f'Clustering: eps {eps}')
        clustering = DBSCAN(eps=eps, min_samples=2).fit(aaa)
        labels = clustering.labels_
        logger.info(len(set(labels)))
        df['cluster']=labels
        logger.info(f'\n{df}')
        df.sort_values(by='cluster',inplace=True)
        df.to_csv(cluster_file,index=False)

def cluster_summary(eps,cluster_file):
    df = pd.read_csv(cluster_file)
    logger.info(len(set(df['cluster'])))
    vc = df['cluster'].value_counts()
    logger.info(f'\n{vc}')

    top_clusters = vc.index.tolist()[:5]
    logger.info(top_clusters)
    for i in top_clusters:
        if i != -1:
            traits = df[df['cluster']==i]['query']
            logger.info(f'{i} {traits}')

if __name__ == "__main__":
    df = read_traits()
    encode_gwas(df)
    run_aaa(df)
    for i in range(1,4):
        cluster_file = f'{output}/trait_list_cluster_eps_{i}.csv'
        cluster(df,i, cluster_file)
        cluster_summary(i, cluster_file)


