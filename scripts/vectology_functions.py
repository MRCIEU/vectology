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

 
