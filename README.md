# empiricaldist

`empiricaldist` is a Python library that provides classes to represent empirical distributions -- that is, distributions based on data rather than mathematical functions.
It includes four equivalent ways to represent a distribution: PMF (Probability Mass Function), CDF (Cumulative Distribution Function), Survival function and Hazard Function.
It provides methods to convert from one representation to the others, and methods to perform a variety of operations.

This library is used extensively in [Think Stats](https://greenteapress.com/wp/think-stats-3e/), [Think Bayes](https://greenteapress.com/wp/think-bayes/), [Elements of Data Science](https://greenteapress.com/wp/elements-of-data-science/), and [Think Complexity](https://greenteapress.com/wp/think-complexity-2e/) -- but it is intended to be a stand-alone library for general use, not just for my books.

[![](https://img.shields.io/pypi/v/empiricaldist.svg)](https://pypi.org/project/empiricaldist/)

## Usage

Here's a quick example of how to use it:

```python
from empiricaldist import Pmf

# Create a PMF object
pmf = Pmf.from_seq([1, 2, 2, 3, 5])

# Make the other representations
cdf = pmf.make_cdf()
surv = cdf.make_surv()
hazard = surv.make_hazard()

# Look up quantities
print(pmf(4))
print(cdf(4))
print(surv(4))
print(hazard(4))

# Cdf and Surv also provide inverse lookups
print(cdf.inverse(0.5))
print(surv.inverse(0.5))
```

For an introduction to the API, you can [read this notebook](https://allendowney.github.io/empiricaldist/dist_demo.html) or [run it on Colab](https://colab.research.google.com/github/AllenDowney/empiricaldist/blob/master/empiricaldist/dist_demo.ipynb).


## Installation

To install `empiricaldist`, use pip:

```bash
pip install empiricaldist
```


## License

`empiricaldist` is available under the BSD 3-clause license. See the [LICENSE](https://github.com/AllenDowney/empiricaldist/blob/master/LICENSE) file for more details.
