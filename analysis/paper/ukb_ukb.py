import pandas as pd
import numpy as np
import requests
import json
import time
import re
import os 
import gzip
import timeit
import matplotlib.pyplot as plt
import difflib
from sklearn.manifold import TSNE
from scripts.vectology_functions import create_aaa_distances, create_pair_distances, embed_text, encode_traits, create_efo_nxo
from loguru import logger
from pandas_profiling import ProfileReport
from skbio.stats.distance import mantel

import seaborn as sns

# Apply the default theme
sns.set_theme()

# globals
ebi_data = 'data/UK_Biobank_master_file.tsv'
efo_nodes = 'data/efo_nodes.csv'
efo_rels = 'data/efo_edges.csv'
nxontology_measure = 'batet'
top_x = 100

# define the models and set some colours
cols = sns.color_palette()
modelData = [
    {'name':'BLUEBERT-EFO','model':'BLUEBERT-EFO','col':cols[0]},
    {'name':'BioBERT','model':'biobert_v1.1_pubmed','col':cols[1]},
    {'name':'BioSentVec','model':'BioSentVec','col':cols[2]},
    {'name':'BlueBERT','model':'NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12','col':cols[3]},
    {'name':'GUSE','model':'GUSEv4','col':cols[4]},    
    {'name':'Spacy','model':'en_core_web_lg','col':cols[5]},
    {'name':'SciSpacy','model':'en_core_sci_lg','col':cols[6]},
    {'name':'Zooma','model':'Zooma','col':cols[7]},
    {'name':'SequenceMatcher','model':'SequenceMatcher','col':cols[8]},
]
palette = {}
for m in modelData:
    palette[m['name']]=m['col']
    
output='output/trait-trait'

tSNE=TSNE(n_components=2)

#create nxontology network of EFO relationships
def create_nx():
    logger.info('Creating nx')
    efo_rel_df=pd.read_csv(efo_rels)
    efo_nx = create_efo_nxo(df=efo_rel_df,child_col='sub',parent_col='obj')
    efo_nx.freeze()
    return efo_nx

def read_ebi():
    # read cleaned EBI data
    ebi_df = pd.read_csv('output/ebi-ukb-cleaned.csv')
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
    return ebi_df,ebi_df_dedup

def create_nx_pairs(ebi_df_dedup,efo_nx):
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

def create_aaa():
    # run all against all for EBI query data
    m = modelData[0]
    for m in modelData:
        name = m['name']
        f1 = f'output/{name}-ebi-encode.npy'
        f2 = f'{output}/{name}-ebi-aaa.npy'
        if os.path.exists(f2):
            logger.info(f'{name} done')
        else:
            if os.path.exists(f1):
                print(m)
                dd = np.load(f1)
                print(len(dd))
                aaa = create_aaa_distances(dd)
                np.save(f2,aaa)
                #print(len(aaa))
            else:
                print(f1,'does not exist')


def write_to_file(model_name,pairwise_data,ebi_df_all,ebi_df_filt):
    print('writing',model_name)
    d=[]
    f = f'{output}/{model_name}-ebi-query-pairwise.tsv.gz'
    if os.path.exists(f):
        print('Already done',f)
    else:
        dedup_id_list=list(ebi_df_filt['mapping_id'])
        dedup_query_list=list(ebi_df_filt['query'])
        dedup_query_list=list(ebi_df_filt['query'])
        #fo = gzip.open(f,'w')
        #fo.write("q1\tefo1\tq2\tefo2\tscore\n".encode('utf-8'))
        ebi_list = list(ebi_df_all['mapping_id'])
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

def create_pairwise(ebi_all,ebi_filt):
    # create pairwise files
    for m in modelData:
        name = m['name']
        f = f'{output}/{name}-ebi-aaa.npy'
        if os.path.exists(f):
            dd = np.load(f'{output}/{name}-ebi-aaa.npy')
            #a=np.load('output/BioSentVec-ebi-aaa.npy')
            logger.info(len(dd))
            #print(len(dd[0]))
            write_to_file(model_name=name,pairwise_data=dd,ebi_df_all=ebi_all,ebi_df_filt=ebi_filt)

def create_pairwise_sequence_matcher(ebi_df_filt):
    f = f'{output}/SequenceMatcher-ebi-query-pairwise.tsv.gz'
    if os.path.exists(f):
        logger.info(f'{f} done')
    else:
        d=[]
        ebi_df_filt_dic = ebi_df_filt.to_dict('records')
        for i in range(0,len(ebi_df_filt_dic)):
            for j in range(i,len(ebi_df_filt_dic)):
                if i != j:
                    distance = difflib.SequenceMatcher(None, ebi_df_filt_dic[i]['query'], ebi_df_filt_dic[j]['query']).ratio()
                    d.append({'q1':ebi_df_filt_dic[i]['query'],'q2':ebi_df_filt_dic[j]['query'],'m1':ebi_df_filt_dic[i]['mapping_id'],'m2':ebi_df_filt_dic[j]['mapping_id'],'score':distance})
        df = pd.DataFrame(d)
        print(df.shape)
        df.drop_duplicates(subset=['q1','q2'],inplace=True)
        print(df.shape)
        df.to_csv(f,sep='\t',compression='gzip',index=False)
                

def create_pairwise_bert_efo(ebi_df):
    # format BERT EFO data
    f = f'{output}/BLUEBERT-EFO-ebi-query-pairwise.tsv.gz'
    if os.path.exists(f):
        logger.info(f'{f} done')
    else:
        be_df = pd.read_csv(f'data/BLUEBERT-EFO-ebi-query-pairwise.csv.gz')
        be_df.rename(columns={'text_1':'q1','text_2':'q2'},inplace=True)
        dedup_query_list=list(ebi_df['query'])
        be_df = be_df[be_df['q1'].isin(dedup_query_list) & be_df['q2'].isin(dedup_query_list)]
        be_df.drop_duplicates(subset=['q1','q2'],inplace=True)
        print(be_df.shape)

        nx_df = pd.read_csv(f'{output}/nx-ebi-pairs-nr.tsv.gz',sep='\t')

        print(nx_df.shape)
        print(be_df.head())
        m = pd.merge(nx_df,be_df,left_on=['q1','q2'],right_on=['q1','q2'],how='left')
        # need to mage values negative to use for spearman analysis against 0-1 scores
        m['score']=m['score']*-1
        logger.info(m.head())
        logger.info(m.shape)
        m.to_csv(f,compression='gzip',index=False,sep='\t')

def com_scores():
    # create df of scores
    com_scores = pd.read_csv(f'{output}/nx-ebi-pairs-nr.tsv.gz',sep='\t')
    com_scores.rename(columns={'score':'nx'},inplace=True)
    #print(com_scores.head())
    # add the distances
    for m in modelData:
        name = m['name']
        f = f'{output}/{name}-ebi-query-pairwise.tsv.gz'
        if os.path.exists(f):
            logger.info(name)
            df = pd.read_csv(f,sep='\t')
            com_scores[name]=df['score']
    logger.info(com_scores.shape)
    logger.info(com_scores.head())
    logger.info(com_scores.describe())
    # drop pairs that have a missing score?
    # com_scores.dropna(inplace=True)
    # or replace with 0?
    com_scores.fillna(0,inplace=True)
    logger.info(com_scores.shape)
    com_scores.to_csv(f'{output}/com_scores.tsv.gz',sep='\t',index=False)

    com_scores.drop(['m1','m2','e1','e2','q1','q2'],axis=1,inplace=True)
    pearson=com_scores.corr(method='pearson')
    spearman=com_scores.corr(method='spearman')
    logger.info(f'\n{pearson}')
    ax=sns.clustermap(spearman)
    ax.savefig(f"{output}/images/spearman.png",dpi=1000)

def compare_models_with_sample(sample,term):
    logger.info(f'Comparing models with {sample}')
    # get pairs and analyse
    models = [x['name'] for x in modelData]
    models.remove('Zooma')
    models.append('nx')
    com_scores = pd.read_csv(f'{output}/com_scores.tsv.gz',sep='\t')
    for model in models:
        logger.info(f'Running {model}')
        com_sample = com_scores[com_scores['q1'].isin(sample) & com_scores['q2'].isin(sample)][['q1','q2',model]]
        # add matching pair
        for i in sample:
            df = pd.DataFrame([[i, i, 1]], columns=['q1','q2',model])
            #print(df)
            com_sample = com_sample.append(df)
            #logger.info(com_sample.shape)
        # add bottom half of triangle
        for i, rows in com_sample.iterrows():
            df = pd.DataFrame([[rows['q2'], rows['q1'], rows[model]]], columns=['q1','q2',model])
            com_sample = com_sample.append(df)
            #logger.info(com_sample.shape)
        if model == 'BLUEBERT-EFO':
            #com_sample[model]=-1*com_sample[model]
            logger.info(f'\n{com_sample}')
        com_sample.drop_duplicates(inplace=True)
        #com_sample['q1']=com_sample['q1'].str[-100:]
        #com_sample['q2']=com_sample['q2'].str[-100:]
        #logger.info(com_sample)
        logger.info(f'\n{com_sample}')
        com_sample = com_sample.pivot(index='q1', columns='q2', values=model)
        com_sample = com_sample.fillna(1)
        
        # 1 minus to work with mantel
        n = 1-com_sample.to_numpy()
        np.save(f'{output}/{model}-{term}.npy',n)

        logger.info(f'\n{com_sample}')
        plt.figure(figsize=(16,7))
        sns.clustermap(
            com_sample,
            cmap='coolwarm'
                    )
        plt.savefig(f"{output}/images/sample-clustermap-{model}-{term}.png",dpi=1000)
        plt.close()
        #ax=sns.clustermap(t)
        #ax.savefig(f"{output}/sample.pdf")

def create_random_queries():
    ran_num = 25
    com_scores = pd.read_csv(f'{output}/com_scores.tsv.gz',sep='\t')
    logger.info(com_scores.head())
    sample = list(com_scores['q1'].sample(n=ran_num,random_state=3))
    logger.info(sample)
    return sample

def term_sample(term):
    df = pd.read_csv(f'{output}/ebi_exact.tsv.gz',sep='\t')
    logger.info(f'\n{df.head()}')
    df = df[df['query'].str.contains(term,flags=re.IGNORECASE, regex=True)]
    logger.info(df.shape)
    logger.info(f'\n{df.head()}')

    sample = list(df['query'])[:30]
    return sample

def manual_samples():
    sample = [
        'Alzheimer s disease',
        'Abnormalities of heart beat',
        'Acute hepatitis A',
        'Sleep disorders',
        'Unspecified dementia',
        'Chronic renal failure',
        'Atopic dermatitis',
        'Crohn s disease',
        'Parkinson s disease',
        'Cushing s syndrome',
        'Secondary parkinsonism',
        'Huntington s disease',
        'Pulmonary valve disorders',
        'Snoring',
        'Pulse rate',
        'Body mass index (BMI)',
        'Whole body fat mass',
        'heart/cardiac problem',
        'sleep apnoea',
        'high cholesterol',
        'Daytime dozing / sleeping (narcolepsy)',
        'heart valve problem/heart murmur',
        'other renal/kidney problem',
        'colitis/not crohns or ulcerative colitis'
    ]

    sample = [
        'Alzheimer s disease',
        'Abnormalities of heart beat',
        'Acute hepatitis A',
        'Sleep disorders',
        'Unspecified dementia',
        'Chronic renal failure',
        'Atopic dermatitis',
        'Crohn s disease',
    ]

    return sample

def run_mantel(term):
    models = [x['name'] for x in modelData]
    models.remove('Zooma')
    models.append('nx')
    d=[]
    for i in range(0,len(models)):
        for j in range(0,len(models)):
            #if i != j:
            m1 = models[i]
            m2 = models[j]
            logger.info(f'{m1} {m2}')
            n1 = np.load(f'{output}/{m1}-{term}.npy')
            #logger.info(n1.shape)
            n2 = np.load(f'{output}/{m2}-{term}.npy')
            #logger.info(n2.shape)
            coeff, p_value, n = mantel(x=n1, y=n2, method='pearson')
            logger.info(f'{coeff} {p_value} {n}')
            d.append({
                'm1':m1,'m2':m2,'coeff':coeff,'p_value':p_value
            })
    df = pd.DataFrame(d)
    logger.info(f'\n{df}')
    df = df.pivot(index='m1', columns='m2', values='coeff')
    logger.info(f'\n{df}')
    plt.figure(figsize=(16,7))
    sns.clustermap(
            df,
            cmap='coolwarm'
            )
    plt.savefig(f"{output}/images/mantel-{term}.png",dpi=1000)
    plt.close()

def sample_checks():
        
    # term = 'neoplasm'
    # sample = term_sample(term=term)
    # compare_models_with_sample(sample=sample,term=term)
    # run_mantel(term)

    # term = 'random'
    # sample = create_random_queries()
    # compare_models_with_sample(sample=sample,term=term)
    # run_mantel(term)
    
    term='manual'
    sample = manual_samples()
    compare_models_with_sample(sample=sample,term='manual')
    run_mantel(term)

def run_all():
    #efo_nx = create_nx()
    #create_nx_pairs_nr(ebi_df,efo_nx)
    create_aaa()
    ebi_all,ebi_filt = read_ebi()
    create_pairwise(ebi_all,ebi_filt)
    create_pairwise_bert_efo(ebi_filt)
    create_pairwise_sequence_matcher(ebi_filt)
    com_scores()
    sample_checks()

def dev():
    #efo_nx = create_nx()
    #ebi_all,ebi_filt = read_ebi()
    #create_nx_pairs_nr(ebi_all,efo_nx)
    #create_aaa()
    #create_pairwise(ebi_all,ebi_filt)
    #create_pairwise_bert_efo(ebi_filt)
    #create_pairwise_sequence_matcher(ebi_filt)
    #com_scores()
    sample_checks()

if __name__ == "__main__":
    dev()
    #run_all()
