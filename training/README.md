# Scripts for training the BlueBERT-EFO model

This directory contains scripts to train the 
BlueBERT-EFO (named `efo-bert` in the training project)
finetuned model primarily with pytorch-lightning and huggingface transformers.

Here for simplicity we assume the environment is a multi-GPU server.
For actual usage a user might need to properly set up the working
environment and then further tweak various parameters.

## Set up

```
conda env create -f environment.yml
conda activate bert-training
```

## Preparing data

- get `"gwas_catalog_v1.0.2-studies_r2020-11-03.tsv"` from
  [GWAS Catalog](https://www.ebi.ac.uk/gwas/downloads)
  and place it under `./data/gwas-catalog/`

```
make \
    get_efo \
    prep_efo \
    prep_efo_gwas
```

## Training

```
cd src
make efo_bert_train
```

Once a final version of the model is trained, use the `efo_bert_save_model.py` to package it, and move the binary
to `./models/efo-bert/` (see file structure).

## Inference (of uk biobank mapping)

- obtain `"ebi-ukb-cleaned.tsv"` from the analysis project
  and place it under `./data/ukbb-test/`

```
cd src
make \
    ukbb_prep_data \
    ukbb_efo_scores \
    ukbb_pairwise_scores \
    ukbb_post_process
```

## Project file structure

A complete structure is expected to look like the following:

```
.
├── cache
│   └── efo_bert
│       ├── model_runs
│       │   ├── version_0
│       │   │   ├── epoch=02-val_loss=0.11.ckpt
│       │   │   ├── hparams.yaml
│       │   │   ├── metrics.csv
│       │   │   └── validation_result_pred_target.csv
│       └── models
│           ├── default.ckpt
│           └── model_0
│               ├── epoch=02-val_loss=0.11.ckpt
│               ├── hparams.yaml
│               ├── metrics.csv
│               └── validation_result_pred_target.csv
├── data
│   ├── efo
│   │   ├── efo_details.json
│   │   ├── efo_details_simplified.json
│   │   ├── epigraphdb_efo_nodes.csv
│   │   └── epigraphdb_efo_rels.csv
│   ├── EFO-UKB-mappings
│   │   ├── ISMB_Mapping_UK_Biobank_to_EFO.pdf
│   │   ├── README.md
│   │   ├── UK_Biobank_master_file.tsv
│   │   ├── ukbiobank_zooma.csv
│   │   └── ukbiobank_zooma.txt
│   ├── gwas-catalog
│   │   ├── gwas_catalog_v1.0.2-studies_r2020-11-03.tsv
│   │   └── README.md
│   └── ukbb-test
│       ├── bluebert_efo_mapping.csv
│       ├── bluebert_efo_rankings.csv
│       ├── ebi-ukb-cleaned.tsv
│       ├── efo_bert_inference.csv
│       ├── README.md
│       ├── ukbb-efo-efo-pairs-full.csv
│       ├── ukbb-efo-efo-pairs-scores.csv
│       ├── ukbb-efo-efo-pairs-text.csv
│       ├── ukbb-efo-efo-pairs-text-scores.csv
│       ├── ukbb-efo-pairs.csv
│       ├── ukbb-trait-trait-pairs.csv
│       └── ukbb-trait-trait-pairs-scores.csv
├── environment.yml
├── __init__.py
├── Makefile
├── models
│   └── efo-bert
│       ├── config.json
│       └── pytorch_model.bin
├── output
│   ├── efo
│   │   ├── efo_combined.csv
│   │   ├── efo_graph.gpickle
│   │   ├── efo_negative_full.csv
│   │   └── efo_positive_full.csv
│   ├── efo-gwas
│   │   ├── combined.csv
│   │   ├── distance.db
│   │   ├── distance_db.csv
│   │   ├── distance_meta.csv
│   │   ├── distance_stage0.csv
│   │   ├── distance_train.csv
│   │   ├── distance_train.db
│   │   ├── distance_val.csv
│   │   ├── distance_val.db
│   │   ├── efo_gwas_graph.gpickle
│   │   ├── negative_full.csv
│   │   ├── node_df.csv
│   │   ├── positive_full.csv
│   │   └── rel_df.csv
│   ├── efo_bert_inference.csv
│   └── ukbb_efo_scores.csv
├── pyproject.toml
├── README.md
├── setup.cfg
├── setup.py
├── src
│   ├── data_processing
│   │   ├── get_efo.py
│   │   ├── __init__.py
│   │   ├── prep_efo_gwas.py
│   │   ├── prep_efo.py
│   │   ├── ukbb_post_process.py
│   │   └── ukbb_prep_data.py
│   ├── efo_bert_inference_ray.py
│   ├── efo_bert_inference.py
│   ├── efo_bert_lr_find.py
│   ├── efo_bert_save_model.py
│   ├── efo_bert_train.py
│   ├── funcs
│   │   ├── callbacks.py
│   │   ├── efo
│   │   │   ├── efo_data_processing.py
│   │   │   ├── efo_bert_data_module.py
│   │   │   └── efo_bert_model.py
│   │   ├── hyperparams.py
│   │   ├── __init__.py
│   │   ├── model_utils.py
│   │   ├── sql_utils.py
│   │   ├── text_utils.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── Makefile
│   ├── settings.py
│   └── wandb
├── src.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
└── tox.ini
```
