import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
from loguru import logger
from scripts.vectology_functions import create_aaa_distances
#data = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})

#print(data)

modelData = [
    {'name':'BERT-EFO','model':'BERT-EFO'},
    {'name':'BioBERT','model':'biobert_v1.1_pubmed'},
    {'name':'BioSentVec','model':'BioSentVec'},
    {'name':'BlueBERT','model':'NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12'},
    {'name':'GUSE','model':'GUSEv4'},    
    {'name':'Spacy','model':'en_core_web_lg'},
    {'name':'SciSpacy','model':'en_core_sci_lg'},
    {'name':'Zooma','model':'Zooma'},
]
output='output/trait-trait' 

ebi_df = pd.read_csv('output/ebi-ukb-cleaned.tsv',sep='\t')
logger.info(ebi_df.head())

def create_aaa():
    # run all against all for EBI query data
    ds = xr.Dataset(coords={"x":ebi_df['mapping_id'],"y":ebi_df['mapping_id']})

    for m in modelData:
        name = m['name']
        f1 = f'output/{name}-ebi-encode.npy'
        f2 = f'{output}/{name}-ebi-aaa.npys'
        data_arrays = []
        if os.path.exists(f2):
            logger.info(f'{name} done')
        else:
            if os.path.exists(f1):
                logger.info(m)
                dd = np.load(f1)
                logger.info(len(dd))
                aaa = create_aaa_distances(dd)
                data = xr.DataArray(aaa,dims=("x","y"),coords={"x":ebi_df['mapping_id'],"y":ebi_df['mapping_id']},name=name)
                data.attrs["long_name"] = name
                data.x.attrs["units"]="cosine distance"
                data.y.attrs["units"]="cosine distance"
                logger.info(data)
                logger.info(data.x.attrs)

                # add datarray to dataset
                ds[name]=data

                # plot - bad idea
                #data.plot()
                #plt.savefig('test.pdf')
                
                #convert to pandas df
                #df = data.to_dataframe()
                #logger.info(df.head())

            else:
                print(f1,'does not exist')

    logger.info(ds)
    # save to file
    ds.to_netcdf(f"{output}/all-aaa.nc")

create_aaa()


#print(xr.__version__)