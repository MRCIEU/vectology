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

#elasticsearch
#connect to local elasticsearch
es = Elasticsearch(['hub-elasticsearch:9200'],timeout=60)
model='BioSentVec'

def create_index(index_name,dim_size):
    es.indices.delete(index=index_name, ignore=[404])
    request_body ={
            "settings":{
                "number_of_shards" : 5,
                "number_of_replicas":0,
                "refresh_interval":-1,
                "index.max_result_window": 100000
            },
            "mappings": {
                "dynamic": "true",
                "_source": {
                  "enabled": "true"
                },
                "properties": {
                  "name": {
                    "type": "text"
                  },
                  "encode_text": {
                    "type": "text"
                  },
                  "full_text": {
                    "type": "text"
                  },
                  "text_vector": {
                    "type": "dense_vector",
                    "dims": dim_size
                  }
                }
              }
        }
    es.indices.create(index = index_name, body = request_body, request_timeout=60)
    
def index_batch(docs,index_name,model):
    encode_text = [doc["encode_text"] for doc in docs]
    text_vectors = embed_text(encode_text,model)
    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = index_name
        request["text_vector"] = text_vectors[i]
        if np.count_nonzero(text_vectors[i])==0:
            print(doc['name'],doc['encode_text'],'returned empty vector so skipping')
            continue
        else:
            requests.append(request)
    bulk(es, requests)
    
def index_data(index_name,df,id_col,text_col,process_col,batch_size=50,model='BioSentVec',dim_size=700):
    filterData=[]
    print("Creating the",index_name,"index.")
    print("First deleting existing index...")
    es.indices.delete(index=index_name, ignore=[404])
    print("Now creating new one...")
    create_index(index_name,dim_size)

    docs = []
    count = 0
    
    for i,row in df.iterrows():
        doc={}
        doc['name'] = row[id_col]
        text = row[text_col]
        process_text = row[process_col]
        filteredRes = filter_text([process_text])[0]['result']
        filterData.append(filteredRes)
        doc['encode_text'] = filteredRes
        doc['full_text'] = text
        docs.append(doc)
        count += 1
        if count % batch_size == 0:
            embedding_start = time.time()
            print(count)
            #print(docs)
            index_batch(docs,index_name,model)
            docs = []
            print("Indexed {} documents.".format(count))
            embedding_time = time.time() - embedding_start
            print("Embedding time: {:.2f} ms".format(embedding_time * 1000))
    if docs:
        index_batch(docs,index_name,model)
        print("Indexed {} documents.".format(count))

    es.indices.refresh(index=index_name,request_timeout=300)
    print("Done indexing.")
    df['filter']=filterData
    return df

def query_record(index_name,query_vector,record_size=100000,search_size=1000,score_min=0):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                #+1 to deal with negative results (script score function must not produce negative scores)
                "source": "cosineSimilarity(params.query_vector, doc['text_vector']) +1",
                "params": {"query_vector": query_vector}
            }
        }
    }
    search_start = time.time()
    response = es.search(
        index=index_name,
        body={
            "size": search_size,
            "query": script_query,
            "_source": {"includes": ["name", "encode_text","full_text"]}
        }
    )
    #search_time = time.time() - search_start
    #print()
    #print("{} total hits.".format(response["hits"]["total"]["value"]))
    #print("search time: {:.2f} ms".format(search_time * 1000))
    results=[]
    for hit in response["hits"]["hits"]:
        #-1 to deal with +1 above
        #print("id: {}, score: {}".format(hit["_id"], hit["_score"] - 1))
        #print(hit["_source"])
        #print()
        #score cutoff
        if hit["_score"]-1>score_min:
            results.append({
                'name':hit["_source"]['name'],
                'score':hit["_score"]-1,
                'encode_text':hit['_source']['encode_text'],
                'full_text':hit['_source']['full_text']
            })
    return results

#get all records from an index
def get_all_records(index_name,record_size=100000):
    print('Getting all records from',index_name)
    response = es.search(
        index=index_name,
        body={
            "size": record_size,
            "query": {"match_all" : {}}
        }
    )
    print("{} Total hits.".format(response["hits"]["total"]["value"]))
    return response

#takes an array of vectors
def create_aaa_distances(vectors=[]):
    print('Creating distances...')
    #https://stackoverflow.com/questions/48838346/how-to-speed-up-computation-of-cosine-similarity-between-set-of-vectors

    print(len(vectors))
    data = np.array(vectors)
    pws = distance.pdist(data, metric='cosine')
    print(len(pws))
    return pws

 
