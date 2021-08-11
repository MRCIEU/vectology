#################################################################################
# GLOBALS                                                                       #
#################################################################################

#################################################################################
# Rules
#################################################################################

## ==== utils ====
__utils__:

## Unit tests
test:
	python -m pytest -v

## ==== codebase ====
__codebase__:

## Lint codebase with flake8
lint:
	flake8; \
	black --diff ./

## Format codebase with black
fmt:
	autoflake --in-place --remove-all-unused-imports --recursive .
	isort -rc .
	black ./

## mypy
type:
	mypy ./

## ==== running the api ====
__run__:

## Start API server
run:
	uvicorn app.main:app --reload


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "$$(tput bold)Params:$$(tput sgr0)"
	@echo "$$(tput setaf 6)API_PORT:$$(tput sgr0) ${API_PORT}"
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
