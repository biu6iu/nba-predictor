ENV ?= nba-predictor

RUN ?= conda run --no-capture-output -n $(ENV)

.DEFAULT_GOAL := help
.PHONY: help setup train dashboard test lint

help: 
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: 
	conda env create -n $(ENV) -f environment.yml || conda env update -n $(ENV) -f environment.yml --prune

train: 
	$(RUN) python train.py

dashboard:
	$(RUN) streamlit run dashboard/app.py

test: 
	$(RUN) pytest

lint: 
	$(RUN) ruff check .
