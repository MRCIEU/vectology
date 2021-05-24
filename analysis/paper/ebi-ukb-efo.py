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

def get_ebi_data():
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
    return ebi_df

if __name__ == "__main__":
    efo_node_df = efo_node_data()
    #ebi_df = get_ebi_data()