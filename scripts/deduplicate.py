import pandas as pd
import xarray as xr
from loguru import logger
from vectology_functions import encode_traits

def read_traits():
    file_name='list_outcomes_exact_matches.csv'
    df = pd.read_csv(file_name)
    logger.info(f'\n{df}')
    return df

def encode_gwas(df):
    df = encode_traits(trait_df=df, col='trait', name='vector' ,model='BioSentVec')
    logger.info(f'\n{df}')

if __name__ == "__main__":
    df = read_traits()
    df = encode_gwas(df)


