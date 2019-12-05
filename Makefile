# Edit setup.py and increment version number

all:
	pandoc -s README.md -o README.rst
	rm dist/*
	python3 setup.py sdist
	twine upload dist/*



