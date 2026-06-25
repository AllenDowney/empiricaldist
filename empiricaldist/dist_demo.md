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

# The empiricaldist API

This notebook documents the most useful features of the `empiricaldist` API.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/empiricaldist/blob/master/empiricaldist/dist_demo.ipynb).

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

## A Pmf is a Series

`empiricaldist` provides `Pmf`, which is a Pandas Series that represents a probability mass function.

```python
from empiricaldist import Pmf
```

You can create a `Pmf` in any of the ways you can create a `Series`, but the most common way is to use `from_seq` to make a `Pmf` from a sequence.

The following is a `Pmf` that represents a six-sided die.

```python
d6 = Pmf.from_seq([1,2,3,4,5,6])
```

By default, the probabilities are normalized to add up to 1.

```python
d6
```

But you can also make an unnormalized `Pmf` if you want to keep track of the counts.

```python
d6 = Pmf.from_seq([1,2,3,4,5,6], normalize=False)
d6
```

Or normalize later (the return value is the prior sum).

```python
d6.normalize()
```

Now the Pmf is normalized.

```python
d6
```

## Properties

In a `Pmf` the index contains the quantities (`qs`) and the values contain the probabilities (`ps`).

These attributes are available as properties that return arrays (same semantics as the Pandas `values` property)

```python
d6.qs
```

```python
d6.ps
```

## Plotting PMFs

`Pmf` provides two plotting functions.  `bar` plots the `Pmf` as a histogram.

```python
def decorate_dice(title):
    """Labels the axes.
    
    title: string
    """
    plt.xlabel('Outcome')
    plt.ylabel('PMF')
    plt.title(title)
```

```python
d6.bar()
decorate_dice('One die')
```

`plot` displays the `Pmf` as a line.

```python
d6.plot()
decorate_dice('One die')
```

## Selection

The bracket operator looks up an outcome and returns its probability.

```python
d6[1]
```

```python
d6[6]
```

Outcomes that are not in the distribution cause a `KeyError`

```
d6[7]
```

You can also use parentheses to look up a quantity and get the corresponding probability.

```python
d6(1)
```

With parentheses, a quantity that is not in the distribution returns `0`, not an error.

```python
d6(7)
```

## Mutation

`Pmf` objects are mutable, but in general the result is not normalized.

```python
d7 = d6.copy()

d7[7] = 1/6
d7
```

```python
d7.sum()
```

```python
d7.normalize()
```

```python
d7.sum()
```

## Statistics

`Pmf` overrides the statistics methods to compute `mean`, `median`, etc.

These functions only work correctly if the `Pmf` is normalized.

```python
d6 = Pmf.from_seq([1,2,3,4,5,6])
```

```python
d6.mean()
```

```python
d6.var()
```

```python
d6.std()
```

## Sampling

`choice` chooses a random values from the Pmf, following the API of `np.random.choice`

```python
d6.choice(size=10)
```

`sample` chooses a random values from the `Pmf`, with replacement.

```python
d6.sample(n=10)
```

## CDFs

`empiricaldist` also provides `Cdf`, which represents a cumulative distribution function.

```python
from empiricaldist import Cdf
```

You can create an empty `Cdf` and then add elements.

Here's a `Cdf` that represents a four-sided die.

```python
d4 = Cdf.from_seq([1,2,3,4])
```

```python
d4
```

## Properties

In a `Cdf` the index contains the quantities (`qs`) and the values contain the probabilities (`ps`).

These attributes are available as properties that return arrays (same semantics as the Pandas `values` property)

```python
d4.qs
```

```python
d4.ps
```

## Displaying CDFs

`Cdf` provides two plotting functions.

`plot` displays the `Cdf` as a line.

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

`step` plots the Cdf as a step function (which is more technically correct).

```python
d4.step()
decorate_dice('One die')
```

## Selection

The bracket operator works as usual.

```python
d4[1]
```

```python
d4[4]
```

## Evaluating CDFs

`Cdf` provides `forward` and `inverse`, which evaluate the CDF and its inverse as functions.

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

You can also call the `Cdf` like a function (which it is).

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
    qs = cdf1.index.union(cdf2.index)
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

## Mutation

`Cdf` objects are mutable, but in general the result is not a valid Cdf.

```python
d4[5] = 1.25
d4
```

```python
d4.normalize()
d4
```

## Statistics

`Cdf` overrides the statistics methods to compute `mean`, `median`, etc.

```python
d6.mean()
```

```python
d6.var()
```

```python
d6.std()
```

## Sampling

`choice` chooses a random values from the Cdf, following the API of `np.random.choice`

```python
d6.choice(size=10)
```

`sample` chooses a random values from the `Cdf`, with replacement.

```python
d6.sample(n=10)
```

## Arithmetic

`Pmf` and `Cdf` provide `add_dist`, which computes the distribution of the sum.


Here's the distribution of the sum of two dice.

```python
d6 = Pmf.from_seq([1,2,3,4,5,6])

twice = d6.add_dist(d6)
twice
```

```python
twice.bar()
decorate_dice('Two dice')
twice.mean()
```

To add a constant to a distribution, you could construct a deterministic `Pmf`

```python
const = Pmf.from_seq([1])
d6.add_dist(const)
```

But `add_dist` also handles constants as a special case:

```python
d6.add_dist(1)
```

Other arithmetic operations are also implemented

```python
d4 = Pmf.from_seq([1,2,3,4])
```

```python
d6.sub_dist(d4)
```

```python
d4.mul_dist(d4)
```

```python
d4.div_dist(d4)
```

## Comparison operators

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
A = Pmf.from_seq([2, 2, 4, 4, 9, 9])
B = Pmf.from_seq([1, 1, 6, 6, 8, 8])
C = Pmf.from_seq([3, 3, 5, 5, 7, 7])
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

## Joint distributions

`Pmf.make_joint` takes two `Pmf` objects and makes their joint distribution, assuming independence.

```python
d4 = Pmf.from_seq(range(1,5))
d4
```

```python
d6 = Pmf.from_seq(range(1,7))
d6
```

```python
joint = Pmf.make_joint(d4, d6)
joint
```

The result is a `Pmf` object that uses a MultiIndex to represent the values.

```python
joint.index
```

If you ask for the `qs`, you get an array of pairs:

```python
joint.qs
```

You can select elements using tuples:

```python
joint[1,1]
```

You can get unnnormalized conditional distributions by selecting on different axes:

```python
Pmf(joint[1])
```

```python
Pmf(joint.loc[:, 1])
```

But `Pmf` also provides `conditional(i, val)` which returns the conditional distribution where variable `i` has the value `val`: 

```python
joint.conditional(0, 1)
```

```python
joint.conditional(1, 1)
```

It also provides `marginal(i)`, which returns the marginal distribution along axis `i`

```python
joint.marginal(0)
```

```python
joint.marginal(1)
```

Here are some ways of iterating through a joint distribution.

```python
for q in joint.qs:
    print(q)
```

```python
for p in joint.ps:
    print(p)
```

```python
for q, p in joint.items():
    print(q, p)
```

```python
for (q1, q2), p in joint.items():
    print(q1, q2, p)
```

Copyright 2021 Allen Downey

BSD 3-clause license: https://opensource.org/licenses/BSD-3-Clause

```python

```
