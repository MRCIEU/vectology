## Methods section for paper

### Setup

```
conda env create -f environment.yml
python -m spacy download en_core_web_lg
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz
```

#### UKB to EFO

```
python -m paper.ukb_efo
```

#### UKB to UKB

```
python -m paper.ukb_ukb
```

#### Deduplication example

```
python -m scripts.deduplicate
```