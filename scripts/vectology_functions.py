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

 
