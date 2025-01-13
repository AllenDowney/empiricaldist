# pip install jupyter-book ghp-import

# Build the Jupyter book version

# copy the notebooks
cp ../empiricaldist/*.ipynb .

# build the HTML version
jb build .

# copy the API docs
cd ..; mkdocs build; cd jb
cp -r ../site/ _build/html/docs

# push it to GitHub
ghp-import -n -p -f _build/html
