.PHONY: clean data lint format requirements


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


release:
	# Make sure you have the latest version setuptool installed
	# pip install --upgrade setuptools wheel twine

	# First edit setup.py and increment version number
	# Get pypi API token from LastPass
	# login with username __token__ and the token as password
	#
	# Push setup.py to GitHub
	# Run tests, black, and push source to GitHub
	rm dist/*
	$(PYTHON_INTERPRETER) setup.py sdist
	twine upload dist/*


docs:
	cd docs && mkdocs build
