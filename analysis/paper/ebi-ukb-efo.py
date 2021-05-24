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
import difflib
from scripts.vectology_functions import create_aaa_distances, create_pair_distances, embed_text, encode_traits, create_efo_nxo
from loguru import logger

import seaborn as sns

# Apply the default theme
sns.set_theme()

# globals
ebi_data = 'data/UK_Biobank_master_file.tsv'
#efo_nodes = 'data/efo-nodes.tsv'
#efo_data = 'data/efo_data.txt.gz'
efo_nodes = 'data/efo_nodes_2021-05-24.csv'
efo_rels = 'data/efo_edges_2021-05-24.csv'
nxontology_measure = 'batet'
top_x = 100
top_nums=[1,2,5,10,20,50,100]

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
output='output/trait-efo'

def efo_node_data():
    # get EFO node data
    df = pd.read_csv(efo_nodes)
    df.rename(columns={'lbl':'efo_label','id':'efo_id'},inplace=True)
    #drop type
    df.drop(['definition','umls'],inplace=True,axis=1)
    df.drop_duplicates(inplace=True)
    logger.info(f'\n{df}')
    logger.info({df.shape})
    return df

def get_ebi_data(efo_node_df):
    f = 'output/ebi-ukb-cleaned.csv'
    if os.path.exists(f):
        logger.info(f'{f} exists')
        ebi_df = pd.read_csv(f)
    else:
        # get the EBI UKB data
        #get ebi data
        #url='https://raw.githubusercontent.com/EBISPOT/EFO-UKB-mappings/master/UK_Biobank_master_file.tsv'
        #ebi_df = pd.read_csv(url,sep='\t')

        ebi_df = pd.read_csv(ebi_data,sep='\t')

        #drop some columns
        ebi_df = ebi_df[['ZOOMA QUERY','MAPPED_TERM_LABEL','MAPPED_TERM_URI','MAPPING_TYPE']]
        ebi_df.rename(columns={'ZOOMA QUERY':'query'},inplace=True)
        logger.info(f'\n{ebi_df.head()}')
        logger.info(ebi_df.shape)

        #create new rows for multiple labels
        #ebi_df = (
        #        ebi_df.assign(label=ebi_df.MAPPED_TERM_LABEL.str.split("\|\|"))
        #        .explode("label")
        #        .reset_index(drop=True).drop('MAPPED_TERM_LABEL',axis=1)
        #    )

        #create new rows for multiple ids
        #ebi_df['MAPPED_TERM_URI']=ebi_df['MAPPED_TERM_URI'].str.replace('\|\|',',')
        #ebi_df['MAPPED_TERM_URI']=ebi_df['MAPPED_TERM_URI'].str.replace('\|',',')
        #ebi_df = (
        #        .explode("id")
        #        .reset_index(drop=True).drop('MAPPED_TERM_URI',axis=1)
        #        ebi_df.assign(id=ebi_df.MAPPED_TERM_URI.str.split(","))
        #    )

        # drop rows with multiple mappings
        ebi_df = ebi_df[~ebi_df['MAPPED_TERM_URI'].str.contains(',',na=False)]
        logger.info(ebi_df.shape)
        ebi_df = ebi_df[~ebi_df['MAPPED_TERM_URI'].str.contains('\|',na=False)]
        logger.info(ebi_df.shape)

        #clean up
        ebi_df['id'] = ebi_df['MAPPED_TERM_URI'].str.strip()

        #remove underscores
        ebi_df['query'] = ebi_df['query'].str.replace('_',' ')

        #drop where query and id are duplicates
        ebi_df.drop_duplicates(subset=['query','id'],inplace=True)
        logger.info(ebi_df.shape)

        #drop nan
        ebi_df.dropna(inplace=True)
        logger.info(ebi_df.shape)
        logger.info(ebi_df.head())

        #drop cases where query and matched text are identical
        logger.info(ebi_df.shape)
        ebi_df=ebi_df[ebi_df['query'].str.lower()!=ebi_df['MAPPED_TERM_LABEL'].str.lower()]
        logger.info(ebi_df.shape)

        # get counts of mapping type
        logger.info(f'\n{ebi_df["MAPPING_TYPE"].value_counts()}')

        # check data against efo nodes

        efo_node_ids = list(efo_node_df['efo_id'])
        ebi_ids = list(ebi_df['id'])
        missing=[]
        matched = []
        for i in ebi_ids:
            match = False
            for s in efo_node_ids:
                if i in s and match == False:
                    matched.append(s)
                    match = True
            if match == False:
                missing.append(i)
        logger.info(f'Missing: {len(missing)} {missing}')

        # remove missing from ukb data
        logger.info(f'\n{ebi_df.head()}')
        logger.info(ebi_df.shape)
        for i in missing:
            ebi_df = ebi_df.drop(ebi_df[ebi_df['id'].str.contains(i)].index)
        ebi_df['full_id'] = matched

        # add index as ID
        ebi_df['mapping_id']=range(1,ebi_df.shape[0]+1)
        logger.info(ebi_df.head())
        logger.info(ebi_df.shape)
        ebi_df.to_csv(f,index=False)
    return ebi_df

def encode_ebi(ebi_df):
    # encode the EBI query terms with Vectology models
    queries = list(ebi_df['query'])
    chunk=10

    vectology_models = ['BioSentVec','BioBERT','BlueBERT']

    for m in modelData:
        name = m['name']
        model = m['model']
        if name in vectology_models: 
            f = f"output/{m['name']}-ebi-encode.npy"
            if os.path.exists(f):
                logger.info(f'{name} done')
            else:
                logger.info(f'Encoding EBI queriues with {model}')
                results=[]
                for i in range(0,len(queries),chunk):
                    if i % 100 == 0:
                        logger.info(i)
                    batch = queries[i:i+chunk]
                    res = embed_text(textList=batch,model=model)
                    for r in res:
                        results.append(r)
                logger.info(f'Results {len(results)}')
                np.save(f,results)

def encode_efo(efo_node_df):
    queries = list(efo_node_df['efo_label'])
    chunk=20

    vectology_models = ['BioSentVec','BioBERT','BlueBERT']

    for m in modelData:
        name = m['name']
        model = m['model']
        if name in vectology_models: 
            f = f"output/{m['name']}-efo-encode.npy"
            if os.path.exists(f):
                logger.info(f'{name} done')
            else:
                logger.info(f'Encoding EFO queriues with {model}')
                results=[]
                for i in range(0,len(queries),chunk):
                    if i % 100 == 0:
                        logger.info(i)
                    batch = queries[i:i+chunk]
                    res = embed_text(textList=batch,model=model)
                    for r in res:
                        results.append(r)
                #logger.info(f'Results {results}')
                np.save(f,results)

# create GUSE encodings for EBI and EFO
def run_guse(ebi_df,efo_node_df):
    f1 = 'output/GUSE-ebi-encode.npy'
    f2 = 'output/GUSE-efo-encode.npy'

    if os.path.exists(f2):
        logger.info(f'{f2} done')
    else:
        # Google Universal Sentence Encoder

        #!pip install  "tensorflow>=2.0.0"
        #!pip install  --upgrade tensorflow-hub

        import tensorflow_hub as hub

        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        guse_ebi_embeddings = embed(ebi_df['query'])
        guse_efo_embeddings = embed(efo_node_df['efo_label'])

        guse_ebi_embeddings_list = []
        for g in guse_ebi_embeddings:
            guse_ebi_embeddings_list.append(g.numpy())
        np.save(f1,guse_ebi_embeddings_list)
        #ebi_df['GUSE']=guse_ebi_embeddings_list

        guse_efo_embeddings_list = []
        for g in guse_efo_embeddings:
            guse_efo_embeddings_list.append(g.numpy())
        np.save(f2,guse_efo_embeddings_list)

def run_spacy(model,name,ebi_df,efo_node_df):
    f1 = f'output/{name}-ebi-encode.npy'
    f2 = f'output/{name}-efo-encode.npy'
    if os.path.exists(f2):
        logger.info(f'{f2} exists')
    else:
        nlp = spacy.load(model)
        ebi_query_docs = list(nlp.pipe(ebi_df['query']))
        efo_label_docs = list(nlp.pipe(efo_node_df['efo_label']))

        spacy_ebi_embeddings_list = []
        for g in ebi_query_docs:
            spacy_ebi_embeddings_list.append(g.vector)
        np.save(f1,spacy_ebi_embeddings_list)

        spacy_efo_embeddings_list = []
        for g in efo_label_docs:
            spacy_efo_embeddings_list.append(g.vector)
        np.save(f2,spacy_efo_embeddings_list)

def run_seq_matcher(ebi_df,efo_node_df):
    f = f'{output}/SequenceMatcher-pairwise.tsv.gz'
    if os.path.exists(f):
        logger.info(f'{f} done')
    else:
        d = []
        ebi_dic = ebi_df.to_dict('records')
        efo_dic = efo_node_df.to_dict('records') 
        for i in range(0,len(ebi_dic)):
            if i % 100 == 0:
                logger.info(i)
            for j in range(0,len(efo_dic)):
                query = ebi_dic[i]['query']
                efo_label = efo_dic[j]['efo_label']
                distance = difflib.SequenceMatcher(None, query, efo_label).ratio()
                #distance = create_edit_distance(query,efo_label)
                d.append({'mapping_id':ebi_dic[i]['mapping_id'],'manual':ebi_dic[i]['full_id'],'prediction':efo_dic[j]['efo_id'],'score':distance})
        df = pd.DataFrame(d)
        logger.info(df.head())
        df.to_csv(f,sep='\t',index=False,compression='gzip')

def create_nx():
    #create nxontology network of EFO relationships
    efo_rel_df=pd.read_csv(efo_rels)
    efo_nx = create_efo_nxo(df=efo_rel_df,child_col='sub',parent_col='obj')
    efo_nx.freeze()
    return efo_nx

def run_pairs(model):
    dd_name = f"{output}/{model}-dd.npy"
    
    #v1 = list(ebi_df[model])
    #v2 = list(efo_df[model])
    if os.path.exists(f'output/{model}-ebi-encode.npy'):
        v1 = np.load(f'output/{model}-ebi-encode.npy')
        v2 = np.load(f'output/{model}-efo-encode.npy')
        
        if os.path.exists(dd_name):
            logger.info(f'{dd_name} already created, loading...')
            with open(dd_name, 'rb') as f:
                dd = np.load(f)
        else:    
            # cosine of lists
            dd = create_pair_distances(v1,v2)
            np.save(dd_name,dd)
        logger.info('done')
        return dd

def write_to_file(model_name,pairwise_data):
    logger.info(f'writing {model_name}')
    f = f'{output}/{model_name}-pairwise.tsv.gz'
    if os.path.exists(f):
        logger.info(f'Already done {f}')
    else:
        fo = gzip.open(f,'w')
        fo.write("mapping_id\tmanual\tprediction\tscore\n".encode('utf-8'))
        ebi_efo_list = list(ebi_df['full_id'])
        efo_list = list(efo_node_df['efo_id'])
        for i in range(0,len(ebi_efo_list)):
            if i % 100 == 0:
                logger.info(i)
            # write to file
            mCount=0
            for j in range(i,len(efo_list)):
                #if i != j:
                score = 1-pairwise_data[i][j]
                fo.write(f"{i+1}\t{ebi_efo_list[i]}\t{efo_list[j]}\t{score}\n".encode('utf-8'))
                mCount+=1

def create_pair_data():
    for m in modelData:
        logger.info(m['name'])
        dd = run_pairs(m['name'])
        if dd is not None:
            #get_top(m['name'],dd)   
            write_to_file(model_name=m['name'],pairwise_data=dd) 
        else:
            logger.info(f'{m["name"]} not done')

# zooma using API
def zoom_api(text):
    zooma_api = 'https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate'
    payload = {
        'propertyValue':text,
        'filter':'required:[none],ontologies:[efo]'
    }
    res = requests.get(zooma_api,params=payload).json()
    if res:
        logger.info(res)
        efo = res[0]['semanticTags'][0]
        confidence = res[0]['confidence']
        return {'query':text,'prediction':efo.strip(),'confidence':confidence.strip()}
    else:
        return {'query':text,'prediction':'NA','confidence':'NA'}
    #zooma_api('Vascular disorders of intestine')

def run_zooma():
    # takes around 3 minutes for 1,000
    f=f'{output}/zooma.tsv'
    if os.path.exists(f):
        logger.info('Zooma done')
    else:
        all_res = []
        count=1
        for q in list(ebi_df['query']):
            res = run_zooma(q)
            res['mapping_id']=count
            all_res.append(res)
            count+=1
        df = pd.DataFrame(all_res)
        #df['manual'] = ebi_df['full_id'][0:5] 
        #df['mapping_id'] = ebi_df['mapping_id'][0:5]
        logger.info(df.head())
        #df = pd.merge(df,ebi_df,left_on='query',right_on='query')
        logger.info(df.shape)
        #df[['mapping_id','manual','prediction','confidence']].to_csv(f,index=False,sep='\t')
        df.to_csv(f,index=False,sep='\t')

def filter_zooma():
    dis_results=[]
    efo_results=[]
    df = pd.read_csv(f'{output}/zooma.tsv',sep='\t')
    # add manual EFO
    df = pd.merge(df,ebi_df[['mapping_id','full_id']],left_on='mapping_id',right_on='mapping_id')
    df.rename(columns={'full_id':'manual'},inplace=True)
    logger.info(df.head())
    for i, row in df.iterrows():
        try:
            res = similarity = efo_nx.similarity(row['prediction'],row['manual']).results()
            dis_results.append(res[nxontology_measure])
        except:
            dis_results.append(0)
            
    logger.info(len(dis_results))
    df['score'] = dis_results
    logger.info(df[df['score']>0.9].shape)
    df.to_csv(f'{output}/Zooma-pairwise-filter.tsv.gz',sep='\t',index=False,compression='gzip')
    #sns.displot(df, x="score",kde=True)

def filter_paiwise_file(model_name):
    logger.info(f'filter_pairwise_file {model_name}')
    f = f"{output}/{model_name}-pairwise-filter.tsv.gz"
    if os.path.exists(f):
        logger.info(f'Already done {model_name}')
        return
    else:
        try:
            df = pd.read_csv(f"{output}/{model_name}-pairwise.tsv.gz",sep='\t')
            logger.info(df.shape)
            df = df.sort_values(by=['score'],ascending=False).groupby('mapping_id').head(top_x)
            # drop duplicates
            df.drop_duplicates(subset=['mapping_id','manual','prediction'],inplace=True)
            logger.info(df.shape)
            df.sort_values(by=['mapping_id','score'],ascending=[True,False]).to_csv(f,sep='\t',index=False,compression='gzip')
        except:
            logger.info(f'Error {model_name}')
            return

def filter_bert():
    df = pd.read_csv(f"data/efo_mk1_inference_top100_no_underscores.csv.gz")
    df_top = df.sort_values(by=['score']).groupby('text_1').head(top_x)
    df_top = pd.merge(df_top,ebi_df[['mapping_id','query','full_id']],left_on='text_1',right_on='query')
    df_top.rename(columns={'full_id':'manual'},inplace=True)
    df_top.drop('query',axis=1,inplace=True)

    #map to predicted EFO
    df_top = pd.merge(df_top,efo_node_df,left_on='text_2',right_on='efo_label')
    df_top.rename(columns={'efo_id':'prediction'},inplace=True)
    df_top.drop('efo_label',axis=1,inplace=True)
    df_top.drop_duplicates(subset=['mapping_id','manual','prediction'],inplace=True)
    logger.info(df_top.head())
    df_top[['mapping_id','manual','prediction','score']].sort_values(by=['mapping_id','score']).to_csv(f'{output}/BERT-EFO-pairwise-filter.tsv.gz',index=False,compression='gzip',sep='\t')
    logger.info(df_top.head())

def get_top_using_pairwise_file(model_name,top_num):
    f = f"{output}/{model_name}-top-{top_num}.tsv.gz"
    if os.path.exists(f):
        logger.info(f'Top done {model_name}')
    else:
        logger.info(f'Reading {model_name}')
        try:
            df = pd.read_csv(f"{output}/{model_name}-pairwise-filter.tsv.gz",sep='\t')
        except:
            logger.info('Data do not exist for',model_name)
            return
        logger.info(df.head())
        logger.info(df.shape)
        #remove duplicates
        df.drop_duplicates(subset=['manual','prediction'],inplace=True)
        top_res = []
        manual_efos = list(ebi_df['full_id'])
        for i in range(0,len(manual_efos)):
        #for i in range(0,10):
            manual_efo = manual_efos[i]
            efo_predictions = df[df['manual']==manual_efo].head(n=top_num)[['prediction','score']]
            #end = time.time()
            # run nxontolog for each
            for j,row in efo_predictions.iterrows():
                predicted_efo = row['prediction']
                score = row['score']
                try:
                    res = efo_nx.similarity(manual_efo,predicted_efo).results()
                    nx_val = res[nxontology_measure]
                except:
                    nx_val = 0
                top_res.append({'mapping_id':i+1,'manual':manual_efo,'prediction':predicted_efo,'score':score,'nx':nx_val})  
        res_df = pd.DataFrame(top_res)
        res_df.to_csv(f,index=False,sep='\t',compression='gzip')

def calc_weighted_average(model_name,top_num,mapping_types):
    f = f"{output}/{model_name}-top-100.tsv.gz"
    print(f)
    res = []
    try:
        df = pd.read_csv(f,sep='\t')
        #print(df.head())
    except:
        print('Data do not exist for',model_name)
        return
    manual_efos = list(ebi_df['full_id'])
    for i in range(0,len(manual_efos)):
        manual_efo = manual_efos[i]
        #filter on type
        mapping_type = ebi_df[ebi_df['full_id']==manual_efo]['MAPPING_TYPE'].values[0]
        #print(mapping_type)
        if mapping_type in mapping_types:
            #print(i,manual_efo)
            nx_scores = list(df[df['manual']==manual_efo].head(n=top_num)['nx'])
            weights = list(reversed(range(1,(len(nx_scores)+1))))
            #print(model_name,manual_efo,nx_scores,weights)
            try:
                weighted_avg = round(np.average( nx_scores, weights = weights),3)
            except:
                weighted_avg = 0
            res.append(weighted_avg)
            #print(nx_scores,weights,weighted_avg)
    print(len(res))
    return res

def run_wa(mapping_types,mapping_name):
    for top_num in top_nums:
        all_res = {}
        for m in modelData:
            if top_num > 1 and m['name'] == 'Zooma':
                print('No data for Zooma')
                continue
            else:
                res = calc_weighted_average(m['name'],top_num,mapping_types)
                if res is not None:
                    all_res[m['name']]=res

        df = pd.DataFrame(all_res)
        df['efo'] = ebi_df['full_id']
        #print(df.head())
        df_melt = pd.melt(df, id_vars=['efo'])
        df_melt.rename(columns={'variable':'Model'},inplace=True)
        #print(df_melt.head())
        #ax = sns.displot(x="value", hue="Model", data=df_melt, kind='kde', cut=0, palette=palette, height=6, cumulative=True,common_norm=False)
        ax = sns.displot(x="value", hue="Model", data=df_melt, kind='kde', cut=0, palette=palette, height=6,common_norm=True)
        #sns_plot = sns.displot(ebi_df, x=f"{model}-nx",kde=True)
        ax.set(xlabel=f'Weighted average of nx', ylabel='Density')
        #ax.set_xscale("log")
        ax.savefig(f"{output}/images/weighted-average-nx-{top_num}-{mapping_name}.png",dpi=1000)

def get_top_hits():
    # get some top counts
    # to do
    # - statify by mapping_type

    ebi_df_exact = ebi_df[ebi_df['MAPPING_TYPE']=='Exact']

    res = []
    for i in modelData:
        fName = f"{output}/{i['name']}-top-100.tsv.gz"
        print(fName)
        df = pd.read_csv(fName,sep='\t')

        df = pd.merge(df,ebi_df[['mapping_id','MAPPING_TYPE']],left_on='mapping_id',right_on='mapping_id')
        df.drop_duplicates(subset=['mapping_id','manual'],inplace=True)   
    
        # filter by mapping_type
        df = df[df['MAPPING_TYPE'].isin(['Exact','Broad','Narrow'])]
        d = dict(df[df['nx']==1]['MAPPING_TYPE'].value_counts())
        d['Model'] = i['name']
        d['Total'] = df[df['nx']==1].shape[0]
        res.append(d)

    print(res)
    res_df = pd.DataFrame(res).sort_values(by='Total',ascending=False)
    print(res_df)
    ax = res_df[['Exact','Broad','Narrow','Model']].plot.bar(stacked=True,figsize=(8,8))
    ax.set_xticklabels(res_df['Model'], rotation=45, ha='right')
    fig = ax.get_figure()
    fig.savefig(f'{output}/images/top-counts-by-type.png',dpi=1000)

def run_high_low(type:str):
    match = []
    print(f'\n##### {type} #####')
    for i in modelData:
        fName = f"{output}/{i['name']}-top-100.tsv.gz"
        print(fName)
        df = pd.read_csv(fName,sep='\t')
        # maybe filter on exact mapping type
        #df = df[df['mapping_id'].isin(exact_mapping_type)]
        df.drop_duplicates(subset=['mapping_id','manual'],inplace=True) 
            
        #print(df.head())
        if type == 'low':
            match.extend(list(set(list(df[df['nx']>0.5]['mapping_id']))))
            missing_df = ebi_df_exact[~ebi_df_exact['mapping_id'].isin(match)]

        elif type == 'high':
            match.extend(list(set(list(df[df['nx']<0.95]['mapping_id']))))
            missing_df = ebi_df_exact[~ebi_df_exact['mapping_id'].isin(match)]

        elif type == 'spread':
            missing_df = ebi_df_exact


    print(missing_df.shape)
    print(missing_df['MAPPING_TYPE'].value_counts())
    #print(missing_df[missing_df['MAPPING_TYPE']=='Exact'].head())

    # add the top prediction from each model to the missing df
    # note, some of them don't have any matches to top 100 - why is that...???
    for i in modelData:
        fName = f"{output}/{i['name']}-top-100.tsv.gz"
        print(fName)
        df = pd.read_csv(fName,sep='\t')
        df.drop_duplicates(subset=['mapping_id','manual'],inplace=True) 
        # get data
        missing_df = pd.merge(df[['mapping_id','prediction','score','nx']],missing_df,left_on='mapping_id',right_on='mapping_id')
        missing_df.rename(columns={'prediction':f"{i['name']}-efo",'score':f"{i['name']}-score",'nx':f"{i['name']}-nx"},inplace=True)

    # get rows with largest range for spread option
    if type == 'spread':
        nx_vals = missing_df.filter(regex=".*-nx",axis=1)
        logger.info(nx_vals)
        nx_vals['std']=nx_vals.std(axis=1)
        logger.info(nx_vals)
        # get top 10 range
        nx_vals_top = nx_vals.sort_values(by='std',ascending=False)
        logger.info(nx_vals_top)
        nx_ind = nx_vals_top.index
        logger.info(nx_ind)
        missing_df = missing_df[missing_df.index.isin(nx_ind)]
        missing_df['std'] = nx_vals['std']
        missing_df = missing_df.sort_values('std',ascending=False)

    # limit all to top 10
    missing_df = missing_df

    # add mean nx
    missing_df['mean-nx']=missing_df.filter(regex='.*-nx',axis=1).mean(numeric_only=True, axis=1)
    # sort by min nx
    if type == 'high':
        missing_df.sort_values(by='mean-nx',ascending=False,inplace=True)
    elif type == 'low':
        missing_df.sort_values(by='mean-nx',ascending=True,inplace=True)
    print(missing_df.head())
    missing_df.to_csv(f'{output}/{type}-predictions.tsv',sep='\t',index=False)

def tidy_up_and_get_rank(df):
    efo_cols = [col for col in df.columns if 'efo' in col]
    #logger.info(efo_cols)
    keep_list = ['query','MAPPED_TERM_LABEL']
    #efo_cols=['SequenceMatcher-efo']
    for e in efo_cols:
        logger.info(e)
        model = e.split('-')[0]
        keep_list.append(model)
        model_efo_name = f'{model}-efo-name'
        logger.info(f'{df[e]} {df[e].map(ebi_dic)}')
        df[model_efo_name] = df[e].map(ebi_dic)
        # get the rank
        top = pd.read_csv(f"{output}/{model}-top-100.tsv.gz",sep='\t')
        logger.info(top)
        rank_vals = []
        for i,row in df.iterrows():
            full_id = row['full_id']
            logger.info(f'#### {full_id}')
            top_match_df = top[top['manual']==full_id].reset_index()
            # find manual efo in top
            match_rank = top_match_df[top_match_df['prediction']==full_id]
            if match_rank.empty:
                match_rank = '>100'
            else:
                logger.info(match_rank.index[0].item())

                match_rank = match_rank.index[0].item()+1
            logger.info(match_rank)
            rank_vals.append(f'{row[model_efo_name]} ({match_rank})')
        df[model] = rank_vals
    df = df[keep_list]
    logger.info(f'\n{df}')
    return df

def create_examples():
    ebi_df = pd.read_csv('output/ebi-ukb-cleaned.tsv',sep='\t')
    ebi_df_exact = ebi_df[ebi_df['MAPPING_TYPE']=='Exact']
    logger.info(ebi_df_exact.shape)

    logger.info(f'\n{ebi_df_exact}')
    ebi_df_exact.drop_duplicates(subset=['query'],inplace=True)
    logger.info(ebi_df_exact.shape)

    run_high_low('low')
    run_high_low('high')
    run_high_low('spread')

    df_low = pd.read_csv(f'{output}/low-predictions.tsv',sep='\t').head(n=10)
    df_low = tidy_up_and_get_rank(df_low)

    df_high = pd.read_csv(f'{output}/high-predictions.tsv',sep='\t').head(n=10)
    df_high = tidy_up_and_get_rank(df_high)

    df_range = pd.read_csv(f'{output}/spread-predictions.tsv',sep='\t').sort_values('std',ascending=False).head(n=10)
    df_range = tidy_up_and_get_rank(df_range)



if __name__ == "__main__":
    efo_node_df = efo_node_data()
    ebi_df = get_ebi_data(efo_node_df)
    encode_ebi(ebi_df)
    encode_efo(efo_node_df)
    run_guse(ebi_df,efo_node_df)
    run_spacy(model="en_core_web_lg",name='Spacy',ebi_df=ebi_df,efo_node_df=efo_node_df)
    run_spacy(model="en_core_sci_lg",name='SciSpacy',ebi_df=ebi_df,efo_node_df=efo_node_df)
    run_seq_matcher(ebi_df,efo_node_df)
    efo_nx = create_nx()
    run_zooma()
    filter_zooma()
    for m in modelData:
        filter_paiwise_file(model_name=m['name'])
        get_top_using_pairwise_file(model_name=m['name'],top_num=100)
    filter_bert()
    run_wa(mapping_types=['Exact','Broad','Narrow'],mapping_name='all')
    run_wa(mapping_types=['Exact'],mapping_name='exact')
    run_wa(mapping_types=['Broad','Narrow'],mapping_name='broad-narrow')
    get_top_hits()
    create_examples()