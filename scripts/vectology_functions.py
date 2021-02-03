import requests
import json
import time
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from scipy.spatial import distance


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
 
