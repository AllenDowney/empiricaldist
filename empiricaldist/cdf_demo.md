---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Implementing CDFs

This notebook outlines the API for `Cdf` objects in the `empiricaldist` library, showing the implementations of many methods.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/empiricaldist/blob/master/empiricaldist/cdf_demo.ipynb).

```python
try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python
import inspect

def psource(obj):
    """Prints the source code for a given object.

    obj: function or method object
    """
    print(inspect.getsource(obj))
```

## Constructor

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/11).

The `Cdf` class inherits its constructor from `pd.Series`.


You can create an empty `Cdf` and then add elements.

Here's a `Cdf` that representat a four-sided die.

```python
from empiricaldist import Cdf

d4 = Cdf()
```

```python
d4[1] = 1
d4[2] = 2
d4[3] = 3
d4[4] = 4
```

```python
d4
```

In a normalized `Cdf`, the last probability is 1.

`normalize` makes that true.  The return value is the total probability before normalizing.

```python
psource(Cdf.normalize)
```

```python
d4.normalize()
```

Now the Cdf is normalized.

```python
d4
```

## Properties

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/2).

In a `Cdf` the index contains the quantities (`qs`) and the values contain the probabilities (`ps`).

These attributes are available as properties that return arrays (same semantics as the Pandas `values` property)

```python
d4.qs
```

```python
d4.ps
```

## Sharing

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/12).

Because `Cdf` is a `Series` you can initialize it with any type `Series.__init__` can handle.

Here's an example with a dictionary.

```python
d = dict(a=1, b=2, c=3)
cdf = Cdf(d)
cdf.normalize()
cdf
```

Here's an example with two lists.

```python
qs = [1,2,3,4]
ps = [0.25, 0.5, 0.75, 1.0]
d4 = Cdf(ps, index=qs)
d4
```

You can copy a `Cdf` like this.

```python
d4_copy = Cdf(d4)
d4_copy
```

However, you have to be careful about sharing.  In this example, the copies share the arrays:

```python
d4.index is d4_copy.index
```

```python
d4.ps is d4_copy.ps
```

You can avoid sharing with `copy=True`

```python
d4_copy = Cdf(d4, copy=True)
d4_copy
```

```python
d4.index is d4_copy.index
```

```python
d4.ps is d4_copy.ps
```

Or by calling `copy` explicitly.

```python
d4_copy = d4.copy()
d4_copy
```

```python
d4.index is d4_copy.index
```

```python
d4.ps is d4_copy.ps
```

## Displaying CDFs

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/13).

`Cdf` provides `_repr_html_`, so it looks good when displayed in a notebook.

```python
psource(Cdf._repr_html_)
```

`Cdf` provides `plot`, which plots the Cdf as a line.

```python
psource(Cdf.plot)
```

```python
def decorate_dice(title):
    """Labels the axes.
    
    title: string
    """
    plt.xlabel('Outcome')
    plt.ylabel('CDF')
    plt.title(title)
```

```python
d4.plot()
decorate_dice('One die')
```

`Cdf` also provides `step`, which plots the Cdf as a step function.

```python
psource(Cdf.step)
```

```python
d4.step()
decorate_dice('One die')
```

## Make Cdf from sequence

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/14).

The following function makes a `Cdf` object from a sequence of values.

```python
psource(Cdf.from_seq)
```

```python
cdf = Cdf.from_seq(list('allen'))
cdf
```

```python
cdf = Cdf.from_seq(np.array([1, 2, 2, 3, 5]))
cdf
```

## Selection

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/15).

`Cdf` inherits [] from Series, so you can look up a quantile and get its cumulative probability.

```python
d4[1]
```

```python
d4[4]
```

`Cdf` objects are mutable, but in general the result is not a valid Cdf.

```python
d4[5] = 1.25
d4
```

```python
d4.normalize()
d4
```

## Evaluating CDFs

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/16).

Evaluating a `Cdf` forward maps from a quantity to its cumulative probability.

```python
d6 = Cdf.from_seq([1,2,3,4,5,6])
```

```python
d6.forward(3)
```

`forward` interpolates, so it works for quantities that are not in the distribution.

```python
d6.forward(3.5)
```

```python
d6.forward(0)
```

```python
d6.forward(7)
```

`__call__` is a synonym for `forward`, so you can call the `Cdf` like a function (which it is).

```python
d6(1.5)
```

`forward` can take an array of quantities, too.

```python
def decorate_cdf(title):
    """Labels the axes.
    
    title: string
    """
    plt.xlabel('Quantity')
    plt.ylabel('CDF')
    plt.title(title)
```

```python
qs = np.linspace(0, 7)
ps = d6(qs)
plt.plot(qs, ps)
decorate_cdf('Forward evaluation')
```

`Cdf` also provides `inverse`, which computes the inverse `Cdf`:

```python
d6.inverse(0.5)
```

`quantile` is a synonym for `inverse`

```python
d6.quantile(0.5)
```

`inverse` and `quantile` work with arrays 

```python
ps = np.linspace(0, 1)
qs = d6.quantile(ps)
plt.plot(qs, ps)
decorate_cdf('Inverse evaluation')
```

These functions provide a simple way to make a Q-Q plot.

Here are two samples from the same distribution.

```python
cdf1 = Cdf.from_seq(np.random.normal(size=100))
cdf2 = Cdf.from_seq(np.random.normal(size=100))

cdf1.plot()
cdf2.plot()
decorate_cdf('Two random samples')
```

Here's how we compute the Q-Q plot.

```python
def qq_plot(cdf1, cdf2):
    """Compute results for a Q-Q plot.
    
    Evaluates the inverse Cdfs for a 
    range of cumulative probabilities.
    
    cdf1: Cdf
    cdf2: Cdf
    
    Returns: tuple of arrays
    """
    ps = np.linspace(0, 1)
    q1 = cdf1.quantile(ps)
    q2 = cdf2.quantile(ps)
    return q1, q2
```

The result is near the identity line, which suggests that the samples are from the same distribution.

```python
q1, q2 = qq_plot(cdf1, cdf2)
plt.plot(q1, q2)
plt.xlabel('Quantity 1')
plt.ylabel('Quantity 2')
plt.title('Q-Q plot');
```

Here's how we compute a P-P plot

```python
def pp_plot(cdf1, cdf2):
    """Compute results for a P-P plot.
    
    Evaluates the Cdfs for all quantities in either Cdf.
    
    cdf1: Cdf
    cdf2: Cdf
    
    Returns: tuple of arrays
    """
    qs = cdf1.index.union(cdf2)
    p1 = cdf1(qs)
    p2 = cdf2(qs)
    return p1, p2
```

And here's what it looks like.

```python
p1, p2 = pp_plot(cdf1, cdf2)
plt.plot(p1, p2)
plt.xlabel('Cdf 1')
plt.ylabel('Cdf 2')
plt.title('P-P plot');
```

## Statistics

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/17).

`Cdf` overrides the statistics methods to compute `mean`, `median`, etc.

```python
psource(Cdf.mean)
```

```python
d6.mean()
```

```python
psource(Cdf.var)
```

```python
d6.var()
```

```python
psource(Cdf.std)
```

```python
d6.std()
```

## Sampling

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/18).

`choice` chooses a random values from the Cdf, following the API of `np.random.choice`

```python
psource(Cdf.choice)
```

```python
d6.choice(size=10)
```

`sample` chooses a random values from the `Cdf`, following the API of `pd.Series.sample`

```python
psource(Cdf.sample)
```

```python
d6.sample(n=10)
```

## Arithmetic

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/9).

`Cdf` provides `add_dist`, which computes the distribution of the sum.

The implementation uses outer products to compute the convolution of the two distributions.

```python
psource(Cdf.add_dist)
```

```python
psource(Cdf.make_same)
```

Here's the distribution of the sum of two dice.

```python
d6 = Cdf.from_seq([1,2,3,4,5,6])

twice = d6.add_dist(d6)
twice
```

```python
twice.step()
decorate_dice('Two dice')
twice.mean()
```

To add a constant to a distribution, you could construct a deterministic `Pmf`

```python
const = Cdf.from_seq([1])
d6.add_dist(const)
```

But `add_dist` also handles constants as a special case:

```python
d6.add_dist(1)
```

Other arithmetic operations are also implemented

```python
d4 = Cdf.from_seq([1,2,3,4])
d6.sub_dist(d4)
```

```python
d4.mul_dist(d4)
```

```python
d4.div_dist(d4)
```

### Comparison operators

`Pmf` implements comparison operators that return probabilities.

You can compare a `Pmf` to a scalar:

```python
d6.lt_dist(3)
```

```python
d4.ge_dist(2)
```

Or compare `Pmf` objects:

```python
d4.gt_dist(d6)
```

```python
d6.le_dist(d4)
```

```python
d4.eq_dist(d6)
```

Interestingly, this way of comparing distributions is [nontransitive]().

```python
A = Cdf.from_seq([2, 2, 4, 4, 9, 9])
B = Cdf.from_seq([1, 1, 6, 6, 8, 8])
C = Cdf.from_seq([3, 3, 5, 5, 7, 7])
```

```python
A.gt_dist(B)
```

```python
B.gt_dist(C)
```

```python
C.gt_dist(A)
```

Copyright 2019 Allen Downey

BSD 3-clause license: https://opensource.org/licenses/BSD-3-Clause
