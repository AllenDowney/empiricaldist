.PHONY: clean data lint format requirements build upload release


PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = empiricaldist
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Format source code with black
format:
	black --config pyproject.toml empiricaldist


## Lint using flake8 and black (use `make format` to do formatting)
lint:
	flake8 empiricaldist
	black --check --config pyproject.toml empiricaldist


tests:
	pytest --no-cov empiricaldist
	cd empiricaldist; pytest --nbmake *.ipynb

test_notebooks:
	jupytext --to ipynb notebooks/test_fit.md notebooks/cricket.md notebooks/chile.md notebooks/president_normality.md
	cd notebooks && pytest --nbmake test_fit.ipynb cricket.ipynb chile.ipynb president_normality.ipynb


## Build sdist and wheel (required for Pyodide / JupyterLite)
build:
	rm -rf dist/*
	$(PYTHON_INTERPRETER) -m pip install -U build
	$(PYTHON_INTERPRETER) -m build

## Upload dist/* to PyPI (uses ~/.pypirc)
upload:
	$(PYTHON_INTERPRETER) -m pip install -U twine
	twine upload dist/*

## Build and upload. Prerequisite: bump version in setup.py; run make lint && make tests
release: build upload


docs:
	cd docs && mkdocs build
