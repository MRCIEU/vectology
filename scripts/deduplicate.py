import pandas as pd
import xarray as xr
import numpy as np
import os
import sknetwork as skn
from sklearn.cluster import DBSCAN

from loguru import logger
from vectology_functions import embed_text, create_aaa_distances

trait_npy = 'trait_list.npy'

def read_traits():
    file_name='list_outcomes_exact_matches.csv'
    df = pd.read_csv(file_name)
    logger.info(f'\n{df}')
    return df

def encode_gwas(df):
    if os.path.exists(trait_npy):
        logger.info(f'{trait_npy} done')
    else:
        queries = list(df["trait"])
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
    dd = np.load(trait_npy)
    aaa = create_aaa_distances(dd)
    # set nan to 0
    where_are_NaNs = aaa[np.isnan(aaa)]=0
    # check for nan
    logger.info(np.any(np.isnan(aaa)))
    logger.info(len(aaa))
    logger.info(aaa.shape)
    
    # xarray
    #data = xr.DataArray(aaa,dims=("x","y"),coords={"x":df['id'],"y":df['id']})
    #data.attrs["long_name"] = 'traits'
    #data.x.attrs["units"]="cosine distance"
    #data.y.attrs["units"]="cosine distance"
    #logger.info(data)
    #logger.info(data.x.attrs)
    return aaa

def cluster(aaa):
    clustering = DBSCAN(eps=10, min_samples=2).fit(aaa)
    labels = clustering.labels_
    logger.info(len(set(labels)))

if __name__ == "__main__":
    df = read_traits()
    encode_gwas(df)
    aaa = run_aaa(df)
    cluster(aaa)


