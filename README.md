# Using language models and ontology topology to perform semantic mapping of traits between biomedical datasets

## Components

Set up [BERT serving API](apis/bert_service/README.md)

TODO: Set up [BioSentVec serving API]()

Set up [main API](apis/vectology_api/README.md)

A complete setup of service components should look like
(a user might need to change the various ports used by components)

- Main Vectology API: `http://localhost:7560`
- BERT model serving API: `http://localhost:8560`

## Analysis

### Setup

```
conda env create -f environment.yml
python -m spacy download en_core_web_lg
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz
```

#### UKB to EFO

Map UK Biobank traits to EFO using various models

```
python -m paper.ukb_efo
```

#### UKB to UKB

Pair-wise UK Biobank trait mappings using various models

```
python -m paper.ukb_ukb
```

#### BlueBERT-EFO analysis results

Set up a [local development environment](training/README.md).

Then obtain the [model package from repo releases](https://github.com/mrcieu/vectology/releases) 
and add to the dev environment (for details see docs above).

Then run the relevant analysis scripts (see the inference section of the docs above).
