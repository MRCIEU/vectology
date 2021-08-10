.DEFAULT_GOAL := help
.PHONY: help

## ======== data processing ========

## data: get efo data
get_efo:
	python -m data_processing.get_efo \
		-j 8

## data: prep efo data
prep_efo:
	python -m data_processing.prep_efo \
		-j 8

## data: prep efo data with gwas annotation
prep_efo_gwas:
	python -m data_processing.prep_efo_gwas \
		-j 8

## ======== efo_bert ========

## efo_bert: lr_find
efo_bert_lr:
	TOKENIZERS_PARALLELISM=true \
	python efo_bert_lr_find.py \
		--fp16 \
		--gpus 1 \
		--max_tokenization_length 32 \
		--train_batch_size 12 \
		--eval_batch_size 12 \
		--train_sample 0.05 \
		--val_sample 0.05 \
		--sql_chunksize 30 \
        -j 4 \
		--lr_find_steps 1000 \
		--lr_find_min_lr 1e-15 \
		--lr_find_max_lr 0.8

## train efo_bert
efo_bert_train:
	TOKENIZERS_PARALLELISM=true \
	python efo_bert_train.py \
		--fp16 \
        --learning_rate 0.000003 \
        --weight_decay 0.01 \
		--max_tokenization_length 32 \
		--train_batch_size 12 \
		--eval_batch_size 12 \
		--train_sample 0.3 \
		--val_sample 0.3 \
		--num_train_epochs 3 \
		--sql_chunksize 30 \
        -j 4 \
        --accelerator ddp \
        --gpus 4

## ======== test example data ========

## ukbb prep data
ukbb_prep_data:
	python -m data_processing.ukbb_prep_data

## ukbb compute scores (ray serve offline)
ukbb_efo_scores:
	python efo_bert_inference_ray.py \
		--data-path ../data/ukbb-test/ukbb-efo-pairs.csv \
		--output-path ../data/ukbb-test/ukbb-efo-scores.csv

## ukbb trait pairwise scores
ukbb_pairwise_scores:
	python efo_bert_inference_ray.py \
		-j 4 \
		--data-path ../data/ukbb-test/ukbb-trait-trait-pairs.csv \
		--output-path ../data/ukbb-test/ukbb-trait-trait-pairs-scores.csv
	python efo_bert_inference_ray.py \
		-j 4 \
		--data-path ../data/ukbb-test/ukbb-efo-efo-pairs-text.csv \
		--output-path ../data/ukbb-test/ukbb-efo-efo-pairs-text-scores.csv

## ukbb post process
ukbb_post_process:
	python -m data_processing.ukbb_post_process

# ==== utils ====

help:
	@echo "$$(tput bold)Params:$$(tput sgr0)"
	@echo
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}'
