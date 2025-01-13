# empiricaldist

`empiricaldist` is a Python library that provides classes to represent empirical distributions -- that is, distributions based on data rather than mathematical functions.
It includes four equivalent ways to represent a distribution: PMF (Probability Mass Function), CDF (Cumulative Distribution Function), Survival function and Hazard Function.
It provides methods to convert from one representation to the others, and methods to perform a variety of operations.

This library is used extensively in [*Think Stats*](https://greenteapress.com/wp/think-stats-3e/), [*Think Bayes*](https://greenteapress.com/wp/think-bayes/), [*Elements of Data Science*](https://greenteapress.com/wp/elements-of-data-science/), and [*Think Complexity*](https://greenteapress.com/wp/think-complexity-2e/) -- but it is intended to be a stand-alone library for general use, not just for my books.

For an introduction to the API, you can [read this notebook](https://allendowney.github.io/empiricaldist/dist_demo.html) or [run it on Colab](https://colab.research.google.com/github/AllenDowney/empiricaldist/blob/master/empiricaldist/dist_demo.ipynb).

[API Reference documentation is here](https://allendowney.github.io/empiricaldist/docs).


## Installation

To install `empiricaldist` with pip from [PyPI](https://pypi.org/project/empiricaldist/):

```bash
pip install empiricaldist
```

Or with conda from [conda-forge](https://anaconda.org/conda-forge/empiricaldist):

```bash
conda install conda-forge::empiricaldist 
```

## License

`empiricaldist` is available under the BSD 3-clause license. See the [LICENSE](https://github.com/AllenDowney/empiricaldist/blob/master/LICENSE) file for more details.
