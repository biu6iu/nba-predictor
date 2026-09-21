ENV ?= nba-predictor

RUN ?= conda run --no-capture-output -n $(ENV)

.DEFAULT_GOAL := help
.PHONY: help setup train dashboard test lint

help:  ## Show available targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Create (or update) the conda env from environment.yml
	conda env create -n $(ENV) -f environment.yml || conda env update -n $(ENV) -f environment.yml --prune

train:  ## Train the model and write artefacts/
	$(RUN) python train.py

dashboard:  ## Launch the Streamlit dashboard
	$(RUN) streamlit run dashboard/app.py

test:  ## Run the test suite 
	$(RUN) pytest

lint:  ## Lint with ruff (needs requirements-dev.txt)
	$(RUN) ruff check .
