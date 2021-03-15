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
    logger.info(ebi_df.head())
    logger.info(ebi_df.shape)

    # limit to Exact
    ebi_df_dedup = ebi_df[ebi_df['MAPPING_TYPE']=='Exact']
    logger.info(ebi_df_dedup.shape)

    #now we need one to one mappings of query and EFO, so drop duplicates
    ebi_df_dedup = ebi_df_dedup.drop_duplicates(subset=['full_id'])
    ebi_df_dedup = ebi_df_dedup.drop_duplicates(subset=['query'])
    logger.info(ebi_df_dedup.shape)
    logger.info(ebi_df_dedup['MAPPING_TYPE'].value_counts())

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

def create_nx_pairs_nr(ebi_df,efo_nx):
    # create nx score for each full_id (non-redundant)
    print(ebi_df.shape)
    f = f"{output}/nx-ebi-pairs-nr.tsv.gz"
    if os.path.exists(f):
        print('nx for ebi done')
    else:
        data = []
        counter=0
        #efos = list(ebi_df_dedup['full_id'])
        #for i,row1 in ebi_df.iterrows():
        for i in range(0,ebi_df.shape[0]):
            m1 = ebi_df.iloc[i]['mapping_id']
            e1 = ebi_df.iloc[i]['full_id']
            q1 = ebi_df.iloc[i]['query']
            if counter % 100 == 0:
                logger.info(counter)    
            for j in range(i,ebi_df.shape[0]):
            #for j,row2 in ebi_df.iterrows():
                m2 = ebi_df.iloc[j]['mapping_id']
                e2 = ebi_df.iloc[j]['full_id']
                q2 = ebi_df.iloc[j]['query']
                pair = sorted([m1,m2])
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
            counter+=1
        logger.info(counter)
        df = pd.DataFrame(data)
        df.to_csv(f,sep='\t',index=False)

def write_to_file(model_name,pairwise_data,ebi_df):
    print('writing',model_name)
    d=[]
    f = f'{output}/{model_name}-ebi-query-pairwise.tsv.gz'
    if os.path.exists(f):
        print('Already done',f)
    else:
        dedup_id_list=list(ebi_df['mapping_id'])
        dedup_query_list=list(ebi_df['query'])
        dedup_query_list=list(ebi_df['query'])
        #fo = gzip.open(f,'w')
        #fo.write("q1\tefo1\tq2\tefo2\tscore\n".encode('utf-8'))
        ebi_list = list(ebi_df['mapping_id'])
        for i in range(0,len(ebi_list)):
            if i % 100 == 0:
                print(i)
            # write to file
            for j in range(i,len(ebi_list)):
                if i != j:
                    if ebi_list[i] in dedup_id_list and ebi_list[j] in dedup_id_list:
                        #if i != j:
                        #print(pairwise_data[i],pairwise_data[j])
                        score = 1-pairwise_data[i][j]
                        
                        # get matching query names
                        query1 = dedup_query_list[dedup_id_list.index(ebi_list[i])]
                        query2 = dedup_query_list[dedup_id_list.index(ebi_list[j])]
                        #print(query1,query2)
                        
                        #fo.write(f"{query1}\t{ebi_list[i]}\t{query2}\t{ebi_list[j]}\t{score}\n".encode('utf-8'))
                        d.append({'q1':query1,'q2':query2,'m1':ebi_list[i],'m2':ebi_list[j],'score':score})
        df = pd.DataFrame(d)
        print(df.shape)
        df.drop_duplicates(subset=['q1','q2'],inplace=True)
        print(df.shape)
        df.to_csv(f,sep='\t',compression='gzip',index=False)

def create_pairwise(ebi_df):
    # create pairwise files
    for m in modelData:
        name = m['name']
        f = f'{output}/{name}-ebi-aaa.npy'
        if os.path.exists(f):
            dd = np.load(f'{output}/{name}-ebi-aaa.npy')
            #a=np.load('output/BioSentVec-ebi-aaa.npy')
            print(len(dd))
            #print(len(dd[0]))
            write_to_file(model_name=name,pairwise_data=dd,ebi_df=ebi_df)

def create_pairwise_bert_efo(ebi_df):
    # format BERT EFO data
    be_df = pd.read_csv(f'data/BERT-EFO-ebi-query-pairwise.csv.gz')
    be_df.rename(columns={'text_1':'q1','text_2':'q2'},inplace=True)
    dedup_query_list=list(ebi_df['query'])
    be_df = be_df[be_df['q1'].isin(dedup_query_list) & be_df['q2'].isin(dedup_query_list)]
    be_df.drop_duplicates(subset=['q1','q2'],inplace=True)
    print(be_df.shape)

    nx_df = pd.read_csv(f'{output}/nx-ebi-pairs-nr.tsv.gz',sep='\t')
    #be_df = pd.read_csv(f'{output}/BERT-EFO-ebi-query-pairwise.tsv.gz',sep='\t')

    print(nx_df.shape)
    print(be_df.head())
    m = pd.merge(nx_df,be_df,left_on=['q1','q2'],right_on=['q1','q2'],how='left')
    m['score']=m['score']*-1
    logger.info(m.head())
    logger.info(m.shape)
    m.to_csv(f'{output}/BERT-EFO-ebi-query-pairwise.tsv.gz',compression='gzip',index=False,sep='\t')

def com_scores():
    # create df of scores
    com_scores = pd.read_csv(f'{output}/nx-ebi-pairs-nr.tsv.gz',sep='\t')
    com_scores.rename(columns={'score':'nx'},inplace=True)
    com_scores.drop(['m1','m2','e1','e2','q1','q2'],axis=1,inplace=True)
    #print(com_scores.head())
    # add the distances
    for m in modelData:
        name = m['name']
        f = f'{output}/{name}-ebi-query-pairwise.tsv.gz'
        if os.path.exists(f):
            logger.info(name)
            df = pd.read_csv(f,sep='\t')
            com_scores[name]=df['score']
    logger.info(com_scores.head())
    pearson=com_scores.corr(method='pearson')
    spearman=com_scores.corr(method='spearman')
    logger.info(f'\n{pearson}')
    ax=sns.clustermap(spearman)
    ax.savefig(f"{output}/spearman.pdf")

if __name__ == "__main__":
    #efo_nx = create_nx()
    #create_nx_pairs_nr(ebi_df,efo_nx)
    ebi_df = read_ebi()
    #create_pairwise(ebi_df)
    create_pairwise_bert_efo(ebi_df)
    com_scores()