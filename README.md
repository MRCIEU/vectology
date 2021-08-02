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

#### Deduplication example

Deduplication example using UK Biobank traits and BioSentVec

```
python -m scripts.deduplicate
```