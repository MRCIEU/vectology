import requests
import json
import time
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from scipy.spatial import distance
from nxontology import NXOntology
from loguru import logger

#vectology api 
#function to get filtered text
def filter_text(textList):
    url='http://vectology-api.mrcieu.ac.uk/preprocess'
    payload={
        "text_list":textList,
        "source":"ukbb"
        }
    #print(payload)
    response = requests.post(url, data=json.dumps(payload))
    #print(response)
    res = response.json()
    #print(res)
    return res

#function to get embedding
def embed_text(textList,model):
    url='http://vectology-api.mrcieu.ac.uk/encode'
    payload={
        "text_list":textList,
        "model_name":model
        }
    #print(payload)
    response = requests.post(url, data=json.dumps(payload))
    #print(response)
    res = response.json()
    #print(res)
    return res['embeddings'] 

#takes an array of vectors
def create_aaa_distances(vectors=[]):
    print('Creating distances...')
    #https://stackoverflow.com/questions/48838346/how-to-speed-up-computation-of-cosine-similarity-between-set-of-vectors

    print(len(vectors))
    data = np.array(vectors)
    pws = distance.pdist(data, metric='cosine')
    #return as square-form distance matrix
    pws = distance.squareform(pws)
    print(len(pws))
    return pws

#takes an array of vectors
def create_pair_distances(v1=[],v2=[]):
    print('Creating distances...')
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist

    print(len(v1),len(v2))
    y = distance.cdist(v1, v2, 'cosine')
    print(len(y))
    return y

# general encode function for df  
def encode_traits(trait_df,col,name,model):

    vectorList=[]
    count = 0
    #loop through 10 rows at a time
    for k,g in trait_df.groupby(np.arange(len(trait_df))//10):
        #get text for embedding
        textList=list(g[col])
        res = embed_text(textList,model)
        
        #add vectors to list
        for i in range(0,len(textList)):
            vectorList.append(res[i])
            
        count+=10
        if count % 1000 == 0:
            print(count,trait_df.shape[0])

    print(len(vectorList),'vectors created')        
    trait_df[name] = vectorList
    return trait_df
 
#create nxontology network 
def create_efo_nxo(df,child_col,parent_col) -> NXOntology:
    nxo = NXOntology()
    
    edges = []
    for i,row in df.iterrows():
        child = row[child_col]
        parent = row[parent_col]
        edges.append((parent,child))
    #print(edges[0:10])
    nxo.graph.add_edges_from(edges)
    return nxo

# create node and edge data from EFO json
def create_efo_data(efo_data_file):
    node_data = []
    edge_data = []
    with open(efo_data_file) as f:
        data = json.load(f)
        for g in data["graphs"]:
            logger.info(f"{len(g['nodes'])} nodes in efo.json")
            for n in g["nodes"]:
                # logger.info(json.dumps(n, indent=4, sort_keys=True))
                efo_id = n["id"]
                if "lbl" in n:
                    efo_lbl = n["lbl"].replace('\n',' ').strip()
                    efo_def = "NA"
                    if "meta" in n:
                        if "definition" in n["meta"]:
                            if "val" in n["meta"]["definition"]:
                                efo_def = n["meta"]["definition"]["val"].replace('\\n',' ').replace('\n',' ').strip()
                    node_data.append(
                        {"id": efo_id, "lbl": efo_lbl, "definition": efo_def}
                    )
            for n in g["edges"]:
                # logger.info(json.dumps(n, indent=4, sort_keys=True))
                edge_data.append(n)
    logger.info(f"{len(node_data)} nodes created")
    node_df = pd.DataFrame(node_data)
    logger.info(node_df.head())
    edge_df = pd.DataFrame(edge_data)
    logger.info(edge_df)
    return node_df, edge_df

def map_trait_to_cat():
    ebi_ukb = 'output/ebi-ukb-cleaned.csv'
    #ukb_coding = 'data/all_field_coding_description.tsv'
    #ukb_cat = 'data/Data_Dictionary_Showcase_updated_02_05_19.csv'
    ebi_df = pd.read_csv(ebi_ukb)
    logger.info(f'\n{ebi_df.head()}')
    logger.info(f'\n{ebi_df.shape}')
    
    #ukb_coding_df = pd.read_csv(ukb_coding,sep='\t',names=['Category','FieldID','Name'])
    #logger.info(f'\n{ukb_coding_df.head()}')
    #ukb_cat_df = pd.read_csv(ukb_cat)
    #logger.info(f'\n{ukb_cat_df.head()}')

    ebi_coding_df = pd.read_csv('data/ukbiobank_zooma.csv')
    ebi_coding_df['PROPERTY_VALUE'] = ebi_coding_df['PROPERTY_VALUE'].str.lower()
    ebi_coding_df.drop_duplicates(subset=['PROPERTY_VALUE'],inplace=True)
    logger.info(f'\n{ebi_coding_df.head()}')

    df = pd.merge(ebi_df,ebi_coding_df[['PROPERTY_VALUE','PROPERTY_TYPE']],left_on='query',right_on='PROPERTY_VALUE',how='left')
    logger.info(df.shape)    
    
    # find which query names are not in category df
    cat_names = ebi_coding_df['PROPERTY_TYPE']
    ebi_df_missing = ebi_df[~ebi_df['query'].isin(cat_names)]
    logger.info(ebi_df_missing) 

    # the missing PROPERTY_VALUE are disease so update
    logger.info(df['PROPERTY_TYPE'].value_counts())
    df.fillna('disease',inplace=True)
    logger.info(df['PROPERTY_TYPE'].value_counts())
    df.to_csv('output/ebi-ukb-cleaned-cat.csv',index=False)


map_trait_to_cat()