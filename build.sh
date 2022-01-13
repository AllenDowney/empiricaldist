#increment the version number in setup.py

# make sure twine is installed
# conda install twine

# remove the old distribution
rm -rf dist

# make the new distribution
python setup.py sdist

# push it to PyPI
twine upload dist/*

# get username and password from Google passwords

