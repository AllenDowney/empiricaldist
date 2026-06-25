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

# Implementing PMFs

This notebook outlines the API for `Pmf` objects in the `empiricaldist` library, showing the implementations of many methods.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/empiricaldist/blob/master/empiricaldist/pmf_demo.ipynb).

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

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/1).

The `Pmf` class inherits its constructor from `pd.Series`.


You can create an empty `Pmf` and then add elements.

Here's a `Pmf` that represents a six-sided die.

```python
from empiricaldist import Pmf

d6 = Pmf()
```

```python
for x in [1,2,3,4,5,6]:
    d6[x] = 1
```

Initially the probabilities don't add up to 1.

```python
d6
```

`normalize` adds up the probabilities and divides through.  The return value is the total probability before normalizing.

```python
psource(Pmf.normalize)
```

```python
d6.normalize()
```

Now the Pmf is normalized.

```python
d6
```

###Properties

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/2).

In a `Pmf` the index contains the quantities (`qs`) and the values contain the probabilities (`ps`).

These attributes are available as properties that return arrays (same semantics as the Pandas `values` property)

```python
d6.qs
```

```python
d6.ps
```

## Sharing

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/3).

Because `Pmf` is a `Series` you can initialize it with any type `Series.__init__` can handle.

Here's an example with a dictionary.

```python
d = dict(a=1, b=2, c=3)
pmf = Pmf(d)
pmf
```

Here's an example with two lists.

```python
qs = [1,2,3,4]
ps = [0.25, 0.25, 0.25, 0.25]
d4 = Pmf(ps, index=qs)
d4
```

You can copy a `Pmf` like this.

```python
d6_copy = Pmf(d6)
d6_copy
```

However, you have to be careful about sharing.  In this example, the copies share the arrays:

```python
d6.index is d6_copy.index
```

```python
d6.ps is d6_copy.ps
```

You can avoid sharing with `copy=True`

```python
d6_copy = Pmf(d6, copy=True)
d6_copy
```

```python
d6.index is d6_copy.index
```

```python
d6.ps is d6_copy.ps
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

## Displaying PMFs

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/4).

`Pmf` provides `_repr_html_`, so it looks good when displayed in a notebook.

```python
psource(Pmf._repr_html_)
```

`Pmf` provides `bar`, which plots the Pmf as a bar chart.

```python
psource(Pmf.bar)
```

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

`Pmf` inherits `plot` from `Series`.

```python
d6.plot()
decorate_dice('One die')
```

<!-- #region -->
## Make Pmf from sequence

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/5).


The following function makes a `Pmf` object from a sequence of values.
<!-- #endregion -->

```python
psource(Pmf.from_seq)
```

```python
pmf = Pmf.from_seq(list('allen'))
pmf
```

```python
pmf = Pmf.from_seq(np.array([1, 2, 2, 3, 5]))
pmf
```

## Selection

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/6).

`Pmf` overrides `__getitem__` to return 0 for values that are not in the distribution.

```python
psource(Pmf.__getitem__)
```

```python
d6[1]
```

```python
d6[6]
```

If you use square brackets to look up a quantity that's not in the `Pmf`, you get a `KeyError`. 

```python
# d6[7]
```

`Pmf` objects are mutable, but in general the result is not normalized.

```python
d7 = d6.copy()
```

```python
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

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/7).

`Pmf` overrides the statistics methods to compute `mean`, `median`, etc.

These functions only work correctly if the `Pmf` is normalized.

```python
psource(Pmf.mean)
```

```python
d6.mean()
```

```python
psource(Pmf.var)
```

```python
d6.var()
```

```python
psource(Pmf.std)
```

```python
d6.std()
```

## Sampling

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/8).

`choice` chooses a random values from the Pmf, following the API of `np.random.choice`

```python
psource(Pmf.choice)
```

```python
d6.choice(size=10)
```

`sample` chooses a random values from the `Pmf`, following the API of `pd.Series.sample`

```python
psource(Pmf.sample)
```

```python
d6.sample(n=10)
```

## Arithmetic

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/9).

`Pmf` provides `add_dist`, which computes the distribution of the sum.

The implementation uses outer products to compute the convolution of the two distributions.

```python
psource(Pmf.add_dist)
```

```python
psource(Pmf.convolve_dist)
```

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
d4.add_dist(const)
```

But `add_dist` also handles constants as a special case:

```python
d4.add_dist(1)
```

Other arithmetic operations are also implemented

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

### Joint distributions

For comments or questions about this section, see [this issue](https://github.com/AllenDowney/EmpyricalDistributions/issues/10).

`Pmf.make_joint` takes two `Pmf` objects and makes their joint distribution, assuming independence.

```python
psource(Pmf.make_joint)
```

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

But `Pmf` also provides `conditional(i, val)` which returns the conditional distribution where the value on level `i` is `val`.

```python
psource(joint.conditional)
```

```python
joint.conditional(0, 1)
```

```python
joint.conditional(1, 1)
```

It also provides `marginal(i)`, which returns the marginal distribution along axis `i`

```python
psource(Pmf.marginal)
```

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

Copyright 2019 Allen Downey

BSD 3-clause license: https://opensource.org/licenses/BSD-3-Clause

```python

```
