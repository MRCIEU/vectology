import pandas as pd
import numpy as np
import requests
import json
import time
import os 
import gzip
import timeit
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scripts.vectology_functions import create_aaa_distances, create_pair_distances, embed_text, encode_traits, create_efo_nxo
from loguru import logger
from pandas_profiling import ProfileReport

import seaborn as sns

# Apply the default theme
sns.set_theme()

# globals
ebi_data = 'data/UK_Biobank_master_file.tsv'
#efo_nodes = 'data/efo-nodes.tsv'
#efo_data = 'data/efo_data.txt.gz'
efo_nodes = 'data/epigraphdb_efo_nodes.csv'
efo_rels = 'data/epigraphdb_efo_rels.csv'
nxontology_measure = 'batet'

modelData = [
    {'name':'BioSentVec','model':'BioSentVec'},
    {'name':'BioBERT','model':'biobert_v1.1_pubmed'},
    {'name':'BlueBERT','model':'NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12'},
    {'name':'GUSE','model':'GUSEv4'},
    {'name':'BERT-EFO','model':'BERT-EFO'},
    {'name':'Zooma','model':'Zooma'}
]

pallete="hls"
output='output/trait-trait'

tSNE=TSNE(n_components=2)

def create_nx():
	#create nxontology network of EFO relationships
    logger.info('Creating nx')
    efo_rel_df=pd.read_csv(efo_rels)
    efo_nx = create_efo_nxo(df=efo_rel_df,child_col='efo.id',parent_col='parent_efo.id')
    efo_nx.freeze()
    return efo_nx

def read_ebi():
    # read cleaned EBI data
    ebi_df = pd.read_csv('output/ebi-ukb-cleaned.tsv',sep='\t')
    print(ebi_df.head())
    print(ebi_df.shape)

    # limit to Exact
    ebi_df_dedup = ebi_df[ebi_df['MAPPING_TYPE']=='Exact']
    print(ebi_df_dedup.shape)

    #now we need one to one mappings of query and EFO, so drop duplicates
    ebi_df_dedup = ebi_df_dedup.drop_duplicates(subset=['full_id'])
    ebi_df_dedup = ebi_df_dedup.drop_duplicates(subset=['query'])
    print(ebi_df_dedup.shape)
    print(ebi_df_dedup['MAPPING_TYPE'].value_counts())

    ebi_df_dedup.to_csv(f'{output}/ebi_exact.tsv.gz',sep='\t')
    #ebi_df_dedup = ebi_df_dedup.head(n=10)
    return ebi_df_dedup

def create_nx_pairs():
    # create nx score for each full_id
    print(ebi_df_dedup.shape)
    f = f"{output}/nx-ebi-pairs.tsv.gz"
    if os.path.exists(f):
        print('nx for ebi done')
    else:
        data = []
        counter=0
        #efos = list(ebi_df_dedup['full_id'])
        for i,row1 in ebi_df_dedup.iterrows():
            m1 = row1['mapping_id']
            e1 = row1['full_id']
            q1 = row1['query']
            if counter % 100 == 0:
                print(counter)    
            for j,row2 in ebi_df_dedup.iterrows():
                m2 = row2['mapping_id']
                e2 = row2['full_id']
                q2 = row2['query']
                #if e1 != e2:
                res = similarity = efo_nx.similarity(e1,e2).results()
                nx_val = res[nxontology_measure]
                    #print(i,e1,e2,nx_val)
                data.append({
                    'm1':m1,
                    'm2':m2,
                    'e1':e1,
                    'e2':e2,
                    'q1':q1,
                    'q2':q2,
                    'nx':nx_val
                })
            counter+=1
        print(counter)
        df = pd.DataFrame(data)
        df.to_csv(f,sep='\t',index=False)
    print('Done')

def create_nx_pairs_nr():
    # create nx score for each full_id (non-redundant)
    print(ebi_df_dedup.shape)
    f = f"{output}/nx-ebi-pairs-nr.tsv.gz"
    if os.path.exists(f):
        print('nx for ebi done')
    else:
        data = []
        pair_check=[]
        counter=0
        #efos = list(ebi_df_dedup['full_id'])
        for i,row1 in ebi_df_dedup.iterrows():
            m1 = row1['mapping_id']
            e1 = row1['full_id']
            q1 = row1['query']
            if counter % 100 == 0:
                print(counter)    
            for j,row2 in ebi_df_dedup.iterrows():
                m2 = row2['mapping_id']
                e2 = row2['full_id']
                q2 = row2['query']
                pair = sorted(m1,m2)
                if pair not in pair_check:
                    if e1 != e2:
                        res = similarity = efo_nx.similarity(e1,e2).results()
                        nx_val = res[nxontology_measure]
                            #print(i,e1,e2,nx_val)
                        data.append({
                            'm1':m1,
                            'm2':m2,
                            'e1':e1,
                            'e2':e2,
                            'q1':q1,
                            'q2':q2,
                            'nx':nx_val
                        })
                pair_check.append(sorted(m1,m2))
            counter+=1
        print(counter)
        df = pd.DataFrame(data)
        df.to_csv(f,sep='\t',index=False)
    print('Done')

if __name__ == "__main__":
    #efo_nx = create_nx()
    ebi_df = read_ebi()