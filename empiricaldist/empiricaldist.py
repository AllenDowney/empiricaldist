"""Classes to represent empirical distributions

https://en.wikipedia.org/wiki/Empirical_distribution_function

Pmf: Represents a Probability Mass Function (PMF).
Cdf: Represents a Cumulative Distribution Function (CDF).
Surv: Represents a Survival Function
Hazard: Represents a Hazard Function
Distribution: Parent class of all distribution representations

Copyright 2019 Allen B. Downey

MIT License: https://opensource.org/licenses/MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d

    :return: modified d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


class Distribution(pd.Series):
    def __init__(self, *args, **kwargs):
        """Initialize a Pmf.

        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        underride(kwargs, name='')
        if args or ('index' in kwargs):
            super().__init__(*args, **kwargs)
        else:
            underride(kwargs, dtype=np.float64)
            super().__init__([], **kwargs)

    @property
    def qs(self):
        """Get the quantities.

        :return: NumPy array
        """
        return self.index.values

    @property
    def ps(self):
        """Get the probabilities.

        :return: NumPy array
        """
        return self.values

    def head(self, n=3):
        """Override Series.head to return a Distribution.

        n: number of rows

        returns: Distribution
        """
        s = super().head(n)
        return self.__class__(s)

    def tail(self, n=3):
        """Override Series.tail to return a Distribution.

        n: number of rows

        returns: Distribution
        """
        s = super().tail(n)
        return self.__class__(s)

    def transform(self, *args, **kwargs):
        """Override to transform the quantities, not the probabilities."""
        qs = self.index.to_series().transform(*args, **kwargs)
        return self.__class__(self.ps, qs, copy=True)

    def _repr_html_(self):
        """Returns an HTML representation of the series.

        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(dict(probs=self))
        return df._repr_html_()

    def __call__(self, qs):
        """Look up quantities.

        qs: quantity or sequence of quantities

        returns: value or array of values
        """
        string_types = (str, bytes, bytearray)

        # if qs is a sequence type, use reindex;
        # otherwise use get
        if hasattr(qs, '__iter__') and not isinstance(qs, string_types):
            s = self.reindex(qs, fill_value=0)
            return s.to_numpy()
        else:
            return self.get(qs, default=0)

    def mean(self):
        """Expected value.

        :return: float
        """
        return self.make_pmf().mean()

    def mode(self, **kwargs):
        """Most common value.

        If multiple quantities have the maximum probability,
        the first maximal quantity is returned.

        :return: float
        """
        return self.make_pmf().mode(**kwargs)

    def var(self):
        """Variance.

        :return: float
        """
        return self.make_pmf().var()

    def std(self):
        """Standard deviation.

        :return: float
        """
        return self.make_pmf().std()

    def median(self):
        """Median (50th percentile).

        There are several definitions of median;
        the one implemented here is just the 50th percentile.

        :return: float
        """
        return self.make_cdf().median()

    def quantile(self, ps, **kwargs):
        """Quantiles.

        Computes the inverse CDF of ps, that is,
        the values that correspond to the given probabilities.

        :return: float
        """
        return self.make_cdf().quantile(ps, **kwargs)

    def credible_interval(self, p):
        """Credible interval containing the given probability.

        p: float 0-1

        :return: array of two quantities
        """
        tail = (1 - p) / 2
        ps = [tail, 1 - tail]
        return self.quantile(ps)

    def choice(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `p` is provided.

        args: same as np.random.choice
        options: same as np.random.choice

        :return: NumPy array
        """
        pmf = self.make_pmf()
        return pmf.choice(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Samples with replacement using probabilities as weights.

        Uses the inverse CDF.

        n: number of values

        :return: NumPy array
        """
        cdf = self.make_cdf()
        return cdf.sample(*args, **kwargs)

    def add_dist(self, x):
        """Distribution of the sum of values drawn from self and x.

        x: Distribution, scalar, or sequence

        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.add_dist(x)
        return self.make_same(res)

    def sub_dist(self, x):
        """Distribution of the diff of values drawn from self and x.

        x: Distribution, scalar, or sequence

        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.sub_dist(x)
        return self.make_same(res)

    def mul_dist(self, x):
        """Distribution of the product of values drawn from self and x.

        x: Distribution, scalar, or sequence

        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.mul_dist(x)
        return self.make_same(res)

    def div_dist(self, x):
        """Distribution of the ratio of values drawn from self and x.

        x: Distribution, scalar, or sequence

        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.div_dist(x)
        return self.make_same(res)

    def pmf_outer(dist1, dist2, ufunc):
        """Computes the outer product of two PMFs.

        dist1: Distribution object
        dist2: Distribution object
        ufunc: function to apply to the qs

        :return: NumPy array
        """
        #TODO: convert other types to Pmf
        pmf1 = dist1
        pmf2 = dist2

        qs = ufunc.outer(pmf1.qs, pmf2.qs)
        ps = np.multiply.outer(pmf1.ps, pmf2.ps)
        return qs * ps

    def gt_dist(self, x):
        """Probability that a value from self is greater than a value from x.

        x: Distribution, scalar, or sequence

        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.gt_dist(x)

    def lt_dist(self, x):
        """Probability that a value from self is less than a value from x.

        x: Distribution, scalar, or sequence

        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.lt_dist(x)

    def ge_dist(self, x):
        """Probability that a value from self is >= than a value from x.

        x: Distribution, scalar, or sequence

        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.ge_dist(x)

    def le_dist(self, x):
        """Probability that a value from self is <= than a value from x.

        x: Distribution, scalar, or sequence

        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.le_dist(x)

    def eq_dist(self, x):
        """Probability that a value from self equals a value from x.

        x: Distribution, scalar, or sequence

        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.eq_dist(x)

    def ne_dist(self, x):
        """Probability that a value from self is <= than a value from x.

        x: Distribution, scalar, or sequence

        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.ne_dist(x)

    def max_dist(self, n):
        """Distribution of the maximum of `n` values from this distribution.

        n: integer

        :return: Distribution, same type as self
        """
        cdf = self.make_cdf().max_dist(n)
        return self.make_same(cdf)

    def min_dist(self, n):
        """Distribution of the minimum of `n` values from this distribution.

        n: integer

        :return: Distribution, same type as self
        """
        cdf = self.make_cdf().min_dist(n)
        return self.make_same(cdf)

    prob_gt = gt_dist
    prob_lt = lt_dist
    prob_ge = ge_dist
    prob_le = le_dist
    prob_eq = eq_dist
    prob_ne = ne_dist

class Pmf(Distribution):
    """Represents a probability Mass Function (PMF)."""

    def copy(self, deep=True):
        """Make a copy.

        :return: new Pmf
        """
        return Pmf(self, copy=deep)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Pmf.

        :return: Pmf
        """
        return self

    # Pmf overrides the arithmetic operations in order
    # to provide fill_value=0 and return a Pmf.

    def add(self, x, **kwargs):
        """Override add to default fill_value to 0.

        x: Distribution or sequence

        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.add(self, x, **kwargs)
        return Pmf(s)

    __add__ = add
    __radd__ = add

    def sub(self, x, **kwargs):
        """Override the - operator to default fill_value to 0.

        x: Distribution or sequence

        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.subtract(self, x, **kwargs)
        return Pmf(s)

    __sub__ = sub
    __rsub__ = sub

    def mul(self, x, **kwargs):
        """Override the * operator to default fill_value to 0.

        x: Distribution or sequence

        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.multiply(self, x, **kwargs)
        return Pmf(s)

    __mul__ = mul
    __rmul__ = mul

    def div(self, x, **kwargs):
        """Override the / operator to default fill_value to 0.

        x: Distribution or sequence

        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.divide(self, x, **kwargs)
        return Pmf(s)

    __div__ = div
    __rdiv__ = div
    __truediv__ = div
    __rtruediv__ = div

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).

        :return: normalizing constant
        """
        total = self.sum()
        self /= total
        return total

    def mean(self):
        """Computes expected value.

        :return: float
        """
        # TODO: error if not normalized
        # TODO: error if the quantities are not numeric
        return np.sum(self.ps * self.qs)

    def mode(self, **kwargs):
        """Most common value.

        If multiple quantities have the maximum probability,
        the first maximal quantity is returned.

        :return: float
        """
        return self.idxmax(**kwargs)

    def var(self):
        """Variance of a PMF.

        :return: float
        """
        m = self.mean()
        d = self.qs - m
        return np.sum(d ** 2 * self.ps)

    def std(self):
        """Standard deviation of a PMF.

        :return: float
        """
        return np.sqrt(self.var())

    def choice(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `p` is provided.

        args: same as np.random.choice
        kwargs: same as np.random.choice

        :return: NumPy array
        """
        underride(kwargs, p=self.ps)
        return np.random.choice(self.qs, *args, **kwargs)

    def add_dist(self, x):
        """Computes the Pmf of the sum of values drawn from self and x.

        x: Distribution, scalar, or sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.add.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs + x)

    def sub_dist(self, x):
        """Computes the Pmf of the diff of values drawn from self and other.

        x: Distribution, scalar, or sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.subtract.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs - x)

    def mul_dist(self, x):
        """Computes the Pmf of the product of values drawn from self and x.

        x: Distribution, scalar, or sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.multiply.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs * x)

    def div_dist(self, x):
        """Computes the Pmf of the ratio of values drawn from self and x.

        x: Distribution, scalar, or sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.divide.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs / x)

    def convolve_dist(self, dist, ufunc):
        """Convolve two distributions.

        dist: Distribution
        ufunc: elementwise function for arrays

        :return: new Pmf
        """
        dist = dist.make_pmf()
        qs = ufunc(self.qs, dist.qs).flatten()
        ps = np.multiply.outer(self.ps, dist.ps).flatten()
        series = pd.Series(ps).groupby(qs).sum()

        return Pmf(series)

    def gt_dist(self, x):
        """Probability that a value from pmf1 is greater than a value from pmf2.

        dist1: Distribution object
        dist2: Distribution object

        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.greater).sum()
        else:
            return self[self.qs > x].sum()

    def lt_dist(self, x):
        """Probability that a value from pmf1 is less than a value from pmf2.

        dist1: Distribution object
        dist2: Distribution object

        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.less).sum()
        else:
            return self[self.qs < x].sum()

    def ge_dist(self, x):
        """Probability that a value from pmf1 is >= than a value from pmf2.

        dist1: Distribution object
        dist2: Distribution object

        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.greater_equal).sum()
        else:
            return self[self.qs >= x].sum()

    def le_dist(self, x):
        """Probability that a value from pmf1 is <= than a value from pmf2.

        dist1: Distribution object
        dist2: Distribution object

        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.less_equal).sum()
        else:
            return self[self.qs <= x].sum()

    def eq_dist(self, x):
        """Probability that a value from pmf1 equals a value from pmf2.

        dist1: Distribution object
        dist2: Distribution object

        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.equal).sum()
        else:
            return self[self.qs == x].sum()

    def ne_dist(self, x):
        """Probability that a value from pmf1 is <= than a value from pmf2.

        dist1: Distribution object
        dist2: Distribution object

        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.not_equal).sum()
        else:
            return self[self.qs != x].sum()

    def pmf_outer(self, dist, ufunc):
        """Computes the outer product of two PMFs.

        dist: Distribution object
        ufunc: function to apply to the qs

        :return: NumPy array
        """
        dist = dist.make_pmf()
        qs = ufunc.outer(self.qs, dist.qs)
        ps = np.multiply.outer(self.ps, dist.ps)
        return qs * ps

    def bar(self, **options):
        """Make a bar plot.

        Note: A previous version of this function use pd.Series.plot.bar,
        but that was a mistake, because that function treats the quantities
        as categorical, even if they are numerical, leading to hilariously
        unexpected results!

        options: passed to plt.bar
        """
        plt.bar(self.qs, self.ps, **options)

    def make_joint(self, other, **options):
        """Make joint distribution (assuming independence).

        :param self:
        :param other:
        :param options: passed to Pmf constructor

        :return: new Pmf
        """
        qs = pd.MultiIndex.from_product([self.qs, other.qs])
        ps = np.multiply.outer(self.ps, other.ps).flatten()
        return Pmf(ps, index=qs, **options)

    def marginal(self, i, name=None):
        """Gets the marginal distribution of the indicated variable.

        i: index of the variable we want
        name: string

        :return: Pmf
        """
        # The following is deprecated now
        # return Pmf(self.sum(level=i))

        # here's the new version
        return Pmf(self.groupby(level=i).sum())

    def conditional(self, i, val, name=None):
        """Gets the conditional distribution of the indicated variable.

        i: index of the variable we're conditioning on
        val: the value the ith variable has to have
        name: string

        :return: Pmf
        """
        pmf = Pmf(self.xs(key=val, level=i), copy=True)
        pmf.normalize()
        return pmf

    def update(self, likelihood, data):
        """Bayesian update.

        likelihood: function that takes (data, hypo) and returns
                    likelihood of data under hypo, P(data|hypo)
        data: in whatever format likelihood understands

        :return: normalizing constant
        """
        for hypo in self.qs:
            self[hypo] *= likelihood(data, hypo)

        return self.normalize()

    def max_prob(self):
        """Value with the highest probability.

        :return: the value with the highest probability
        """
        return self.idxmax()

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Pmf.

        :return: Cdf
        """
        normalize = kwargs.pop('normalize', False)

        cumulative = np.cumsum(self)
        cdf = Cdf(cumulative, self.index.copy(), **kwargs)

        if normalize:
            cdf.normalize()

        return cdf

    def make_surv(self, **kwargs):
        """Make a Surv from the Pmf.

        :return: Surv
        """
        cdf = self.make_cdf()
        return cdf.make_surv(**kwargs)

    def make_hazard(self, normalize=False, **kwargs):
        """Make a Hazard from the Pmf.

        :return: Hazard
        """
        surv = self.make_surv()
        haz = Hazard(self / (self + surv), **kwargs)
        haz.total = getattr(surv, 'total', 1.0)
        if normalize:
            self.normalize()
        return haz

    def make_same(self, dist):
        """Convert the given dist to Pmf

        :param dist:
        :return: Pmf
        """
        return dist.make_pmf()

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, ascending=True,
                 dropna=True, na_position='last', **options):
        """Make a PMF from a sequence of values.

        seq: iterable
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        ascending: whether to sort in ascending order, default True
        dropna: whether to drop NaN values, default True
        na_position: If ‘first’ puts NaNs at the beginning,
                        ‘last’ puts NaNs at the end.
        options: passed to the pd.Series constructor


        NOTE: In the current implementation, `from_seq` sorts numerical
           quantities whether you want to or not.  If keeping
           the order of the elements is important, let me know and
           I'll rethink the implementation

        :return: Pmf object
        """
        # compute the value counts
        series = pd.Series(seq).value_counts(normalize=normalize,
                                             sort=False,
                                             dropna=dropna)
        # make the result a Pmf
        # (since we just made a fresh Series, there is no reason to copy it)
        options['copy'] = False
        pmf = Pmf(series, **options)

        # sort in place, if desired
        if sort:
            pmf.sort_index(inplace=True,
                           ascending=ascending,
                           na_position=na_position)

        return pmf


class Cdf(Distribution):
    """Represents a Cumulative Distribution Function (CDF)."""

    def copy(self, deep=True):
        """Make a copy.

        :return: new Cdf
        """
        return Cdf(self, copy=deep)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a CDF from a sequence of values.

        seq: iterable
        normalize: whether to normalize the Cdf, default True
        sort: whether to sort the Cdf by values, default True
        options: passed to the pd.Series constructor

        :return: CDF object
        """
        # if normalize==True, normalize AFTER making the Cdf
        # so the last element is exactly 1.0
        pmf = Pmf.from_seq(seq, normalize=False, sort=sort, **options)
        return pmf.make_cdf(normalize=normalize)

    def step(self, **options):
        """Plot the Cdf as a step function.

        :param options: passed to pd.Series.plot

        :return:
        """
        underride(options, drawstyle="steps-post")
        self.plot(**options)

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).

        :return: normalizing constant
        """
        total = self.ps[-1]
        self /= total
        return total

    @property
    def forward(self, **kwargs):
        """Compute the forward Cdf

        :param kwargs: keyword arguments passed to interp1d

        :return interpolation function from qs to ps
        """

        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(0, 1),
        )

        interp = interp1d(self.qs, self.ps, **kwargs)
        return interp

    @property
    def inverse(self, **kwargs):
        """Compute the inverse Cdf

        :param kwargs: keyword arguments passed to interp1d

        :return: interpolation function from ps to qs
        """
        underride(
            kwargs,
            kind="next",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(self.qs[0], np.nan),
        )

        interp = interp1d(self.ps, self.qs, **kwargs)
        return interp

    # calling a Cdf like a function does forward lookup
    __call__ = forward

    # quantile is the same as an inverse lookup
    quantile = inverse

    def median(self):
        """Median (50th percentile).

        :return: float
        """
        return self.quantile(0.5)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Cdf.

        :param normalize: Boolean, whether to normalize the Pmf

        :return: Pmf
        """
        #TODO: check for consistent behavior of copy flag for all make_x
        normalize = kwargs.pop('normalize', False)

        diff = np.diff(self, prepend=0)
        pmf = Pmf(diff, index=self.index.copy(), **kwargs)
        if normalize:
            pmf.normalize()
        return pmf

    def make_surv(self, **kwargs):
        """Make a Surv object from the Cdf.

        :return: Surv object
        """
        normalize = kwargs.pop('normalize', False)
        total = self.ps[-1]
        surv = Surv(total - self, **kwargs)
        surv.total = total
        if normalize:
            self.normalize()
        return surv

    def make_hazard(self, **kwargs):
        """Make a Hazard from the Cdf.

        :return: Hazard
        """
        pmf = self.make_pmf()
        surv = self.make_surv()
        haz = Hazard(pmf / (pmf + surv), **kwargs)
        haz.total = getattr(surv, 'total', 1.0)
        return haz

    def make_same(self, dist):
        """Convert the given dist to Cdf

        :param dist:
        :return: Cdf
        """
        return dist.make_cdf()

    def sample(self, n=1):
        """Samples with replacement using probabilities as weights.

        n: number of values

        :return: NumPy array
        """
        ps = np.random.random(n)
        return self.inverse(ps)

    def max_dist(self, n):
        """Distribution of the maximum of `n` values from this distribution.

        n: integer

        :return: Cdf
        """
        ps = self**n
        return Cdf(ps, self.index.copy())

    def min_dist(self, n):
        """Distribution of the minimum of `n` values from this distribution.

        n: integer

        :return: Cdf
        """
        ps = 1 - (1 - self)**n
        return Cdf(ps, self.index.copy())

class Surv(Distribution):
    """Represents a survival function (complementary CDF)."""

    def copy(self, deep=True):
        """Make a copy.

        :return: new Surv
        """
        return Surv(self, copy=deep)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a Surv from a sequence of values.

        seq: iterable
        normalize: whether to normalize the Surv, default True
        sort: whether to sort the Surv by values, default True
        options: passed to the pd.Series constructor

        :return: Surv object
        """
        cdf = Cdf.from_seq(seq, normalize=normalize, sort=sort, **options)
        return cdf.make_surv()

    def step(self, **options):
        """Plot the Surv as a step function.

        :param options: passed to pd.Series.plot
        :return:
        """
        underride(options, drawstyle="steps-post")
        self.plot(**options)

    def normalize(self):
        """Normalize the survival function (modifies self).

        :return: normalizing constant
        """
        old_total = getattr(self, 'total', 1.0)
        self.ps /= old_total
        self.total = 1.0
        return old_total

    @property
    def forward(self, **kwargs):
        """Compute the forward survival function

        :param kwargs: keyword arguments passed to interp1d

        :return array of probabilities
        """
        total = getattr(self, 'total', 1.0)
        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(total, 0),
        )
        interp = interp1d(self.qs, self.ps, **kwargs)
        return interp

    @property
    def inverse(self, **kwargs):
        """Compute the inverse survival function

        :param kwargs: keyword arguments passed to interp1d

        :return: interpolation function from ps to qs
        """
        total = getattr(self, 'total', 1.0)
        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(np.nan, np.nan),
        )
        # sort in descending order
        # I don't remember why
        rev = self.sort_values()

        # If the reversed Surv doesn't get all the way to total
        # add a fake entry at -inf
        if rev.iloc[-1] != total:
            rev[-np.inf] = total

        interp = interp1d(rev, rev.index, **kwargs)
        return interp

    # calling a Surv like a function does forward lookup
    __call__ = forward

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Surv.

        :return: Cdf
        """
        normalize = kwargs.pop('normalize', False)
        total = getattr(self, 'total', 1.0)
        cdf = Cdf(total - self, **kwargs)
        if normalize:
            cdf.normalize()
        return cdf

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Surv.

        :return: Pmf
        """
        cdf = self.make_cdf()
        pmf = cdf.make_pmf(**kwargs)
        return pmf

    def make_hazard(self, **kwargs):
        """Make a Hazard from the Surv.

        :return: Hazard
        """
        pmf = self.make_pmf()
        at_risk = self + pmf
        haz = Hazard(pmf / at_risk, **kwargs)
        haz.total = getattr(self, 'total', 1.0)
        haz.name = self.name
        return haz

    def make_same(self, dist):
        """Convert the given dist to Surv

        :param dist:
        :return: Surv
        """
        return dist.make_surv()


class Hazard(Distribution):
    """Represents a Hazard function."""

    def copy(self, deep=True):
        """Make a copy.

        :return: new Pmf
        """
        return Hazard(self, copy=deep)

    # Hazard inherits __call__ from Distribution

    def normalize(self):
        """Normalize the hazard function (modifies self).

        :return: normalizing constant
        """
        old_total = getattr(self, 'total', 1.0)
        self.total = 1.0
        return old_total

    def bar(self, **options):
        """Make a bar plot.

        Note: A previous version of this function use pd.Series.plot.bar,
        but that was a mistake, because that function treats the quantities
        as categorical, even if they are numerical, leading to hilariously
        unexpected results!

        options: passed to plt.bar
        """
        plt.bar(self.qs, self.ps, **options)

    def make_surv(self, **kwargs):
        """Make a Surv from the Hazard.

        :return: Surv
        """
        normalize = kwargs.pop('normalize', False)
        ps = (1 - self).cumprod()
        total = getattr(self, 'total', 1.0)
        surv = Surv(ps * total, **kwargs)
        surv.total = total

        if normalize:
            surv.normalize()
        return surv

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Hazard.

        :return: Cdf
        """
        return self.make_surv().make_cdf(**kwargs)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Hazard.

        :return: Pmf
        """
        return self.make_surv().make_cdf().make_pmf(**kwargs)

    def make_same(self, dist):
        """Convert the given dist to Hazard.

        :param dist:
        :return: Hazard
        """
        return dist.make_hazard()

    @staticmethod
    def from_seq(seq, **kwargs):
        """Make a Hazard from a sequence of values.

        seq: iterable
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        kwargs: passed to the pd.Series constructor

        :return: Hazard object
        """
        pmf = Pmf.from_seq(seq, **kwargs)
        return pmf.make_hazard()
