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

# Implementing tail distributions

This notebook outlines the API for `TailDist` objects in the `empiricaldist` library.

A `TailDist` represents the tail distribution **P(X ≥ x)**. It is similar to a survival function, but a `Surv` object represents **P(X > x)**. The difference matters when a distribution has point masses, as empirical distributions often do.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/empiricaldist/blob/master/empiricaldist/tail_demo.ipynb).

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

## Tail vs survival

For a discrete distribution with sorted support qᵢ:

- **Tail:** T(qᵢ) = P(X ≥ qᵢ)
- **Survival:** S(qᵢ) = P(X > qᵢ)

At each support point, T(x) = S(x) + P(X = x). Equivalently, S(qᵢ) = T(qᵢ₊₁), with S at the last support point equal to 0.

We'll use this sequence as a running example:

```python
t = [1, 2, 2, 3, 5]
```

```python
from empiricaldist import Pmf, Surv, TailDist

pmf = Pmf.from_seq(t)
surv = Surv.from_seq(t)
tail = TailDist.from_seq(t)
```

The PMF gives the probability of each quantity:

```python
pmf
```

The survival function gives P(X > x):

```python
surv
```

The tail distribution gives P(X ≥ x):

```python
tail
```

Notice that the tail at each support point includes the point mass at that quantity. For example, P(X ≥ 5) = 0.2 because all of the probability at 5 is included.

## Constructor

The `TailDist` class inherits its constructor from `pd.Series`.

You can build a tail distribution from a PMF by adding the survival function and the PMF, then wrapping the result in a `TailDist`:

```python
ps = pmf.make_surv() + pmf
tail2 = TailDist(ps)
tail2.normalize()
tail2.iloc[0] = 1.0
tail2
```

Or use `from_seq`, which does this for you:

```python
psource(TailDist.from_seq)
```

```python
tail = TailDist.from_seq(t)
tail
```

Other distribution classes provide `make_tail`, which returns the same result:

```python
psource(Pmf.make_tail)
```

```python
pmf.make_tail()
```

```python
from empiricaldist import Cdf

Cdf.from_seq(t).make_tail()
```

```python
surv.make_tail()
```

## Properties

In a `TailDist` the index contains the quantities (`qs`) and the values contain the tail probabilities (`ps`).

```python
tail.qs
```

```python
tail.ps
```

## Displaying tail distributions

`TailDist` provides `plot` and `step`, which draw the tail as a line or step function.

```python
def decorate_tail(title):
    """Labels the axes.

    title: string
    """
    plt.xlabel('Quantity')
    plt.ylabel('P(X ≥ x)')
    plt.title(title)
```

```python
tail.step()
decorate_tail('Tail distribution')
```

Compare the tail with the survival function on the same axes:

```python
tail.step(label='tail')
surv.step(label='surv')
decorate_tail('Tail vs survival')
plt.legend();
```

## Evaluating tail distributions

Evaluating a `TailDist` forward maps from a quantity to P(X ≥ x).

```python
tail(1)
```

```python
tail(2)
```

```python
tail(3.5)
```

`__call__` is a synonym for `forward`, so you can call a `TailDist` like a function.

```python
tail(5)
```

`inverse` maps from a tail probability to a quantity:

```python
tail.inverse(1)
```

```python
tail.inverse(0.8)
```

```python
tail.inverse(0.2)
```

`quantile` is a synonym for `inverse`.

```python
tail.quantile(0.4)
```

## Converting to other representations

`TailDist` provides the same conversion methods as the other distribution classes.

### Surv

`make_surv` converts a tail distribution to a survival function using

    surv[i] = tail[i + 1]

with `surv[-1] = 0`.

```python
psource(TailDist.make_surv)
```

```python
surv2 = tail.make_surv()
surv2
```

The result matches `Surv.from_seq`:

```python
np.allclose(surv2.ps, surv.ps)
```

### Pmf

`make_pmf` recovers the PMF by differencing adjacent tail probabilities.

```python
psource(TailDist.make_pmf)
```

```python
pmf2 = tail.make_pmf()
pmf2
```

```python
np.allclose(pmf2.ps, pmf.ps)
```

### Cdf

`make_cdf` goes through the PMF.

```python
from empiricaldist import Cdf

cdf = Cdf.from_seq(t)
cdf2 = tail.make_cdf()
np.allclose(cdf2.ps, cdf.ps)
```

Round-trip conversions preserve the distribution:

```python
tail3 = pmf.make_tail()
np.allclose(tail3.ps, tail.ps)
```

```python
pmf3 = tail.make_surv().make_pmf()
np.allclose(pmf3.ps, pmf.ps)
```

## Normalize

`normalize` divides through by the total tail mass at the leftmost support point.

```python
psource(TailDist.normalize)
```

```python
tail = TailDist.from_seq(t, normalize=False)
tail
```

```python
total = tail.normalize()
total
```

```python
tail
```

Copyright 2026 Allen Downey

BSD 3-clause license: https://opensource.org/licenses/BSD-3-Clause
