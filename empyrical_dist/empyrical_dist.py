"""

Pmf: Represents a Probability Mass Function (PMF).
Cdf: Represents a Cumulative Distribution Function (CDF).

Copyright 2019 Allen B. Downey

MIT License: https://opensource.org/licenses/MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    def _repr_html_(self):
        """Returns an HTML representation of the series.

        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(dict(probs=self))
        return df._repr_html_()

    def add_dist(self, x):
        """Computes the Pmf of the sum of values drawn from self and x.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.add.outer)
        else:
            return Pmf(self.ps, index=self.qs + x)

    def sub_dist(self, x):
        """Computes the Pmf of the diff of values drawn from self and other.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.subtract.outer)
        else:
            return Pmf(self.ps, index=self.qs - x)

    def mul_dist(self, x):
        """Computes the Pmf of the product of values drawn from self and x.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.multiply.outer)
        else:
            return Pmf(self.ps, index=self.qs * x)

    def div_dist(self, x):
        """Computes the Pmf of the ratio of values drawn from self and x.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.divide.outer)
        else:
            return Pmf(self.ps, index=self.qs / x)

    def convolve_dist(dist1, dist2, ufunc):
        """Convolve two distributions.

        dist1: Distribution
        dist2: Distribution
        ufunc: elementwise function for arrays

        :return: new instance of the type of dist1
        """
        #TODO: Convert to Pmf
        pmf1 = dist1
        pmf2 = dist2

        qs = ufunc(pmf1.qs, pmf2.qs).flatten()
        ps = np.multiply.outer(pmf1.ps, pmf2.ps).flatten()
        series = pd.Series(ps).groupby(qs).sum()

        # TODO: return correct type
        return Pmf(series)

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


class Pmf(Distribution):
    """Represents a probability Mass Function (PMF)."""

    def copy(self, deep=True):
        """Make a copy.

        :return: new Pmf
        """
        return Pmf(self, copy=deep)

    def __getitem__(self, qs):
        """Look up qs and return ps."""
        try:
            return super().__getitem__(qs)
        except (KeyError, ValueError, IndexError):
            return 0

    # You can also call a Pmf like a function
    __call__ = __getitem__

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

    def median(self):
        """Median (50th percentile).

        :return: float
        """
        return self.quantile(0.5)

    def quantile(self, ps, **kwargs):
        """Quantiles.

        Computes the inverse CDF of ps, that is,
        the values that correspond to the given probabilities.

        :return: float
        """
        return self.make_cdf().quantile(ps, **kwargs)

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

    def sample(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `weights` is provided.

        This function returns an array containing a sample of the quantities in this Pmf,
        which is different from Series.sample, which returns a Series with a sample of
        the rows in the original Series.

        args: same as Series.sample
        options: same as Series.sample

        :return: NumPy array
        """
        series = pd.Series(self.qs)
        underride(kwargs, weights=self.ps)
        sample = series.sample(*args, **kwargs)
        return sample.values

    def plot(self, **options):
        """Plot the Pmf as a line.

        :param options: passed to plt.plot
        :return:
        """
        underride(options, label=self.name, color="C1")
        plt.plot(self.qs, self.ps, **options)

    def bar(self, **options):
        """Makes a bar plot.

        options: passed to plt.bar
        """
        underride(options, label=self.name, color="C1")
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
        # TODO: rewrite this using MultiIndex operations
        pmf = Pmf(name=name)
        for vs, p in self.items():
            pmf[vs[i]] += p
        return pmf

    def conditional(self, i, j, val, name=None):
        """Gets the conditional distribution of the indicated variable.

        Distribution of vs[i], conditioned on vs[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have
        name: string

        :return: Pmf
        """
        # TODO: rewrite this using MultiIndex operations
        pmf = Pmf(name=name)
        for vs, p in self.items():
            if vs[j] == val:
                pmf[vs[i]] += p

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

        It can be good to normalize the Cdf even if the Pmf was normalized,
        to guarantee that the last element is 1.

        :return: Cdf
        """
        normalize = kwargs.pop('normalize', False)
        cdf = Cdf(self.cumsum(), **kwargs)
        if normalize:
            cdf.normalize()
        return cdf

    def make_surv(self, **kwargs):
        """Make a Surv object from the Pmf.

        :return: Cdf
        """
        cdf = self.make_cdf()
        return cdf.make_surv(**kwargs)

    def make_hazard(self, **kwargs):
        """Make a Hazard object from the Pmf.

        :return: Cdf
        """
        cdf = self.make_cdf()
        return cdf.make_surv().make_hazard(**kwargs)

    def quantile(self, ps):
        """Quantities corresponding to given probabilities.

        ps: sequence of probabilities

        return: sequence of quantities
        """
        cdf = self.make_cdf()
        return cdf.quantile(ps)

    def credible_interval(self, p):
        """Credible interval containing the given probability.

        p: float 0-1

        :return: array of two quantities
        """
        tail = (1 - p) / 2
        ps = [tail, 1 - tail]
        return self.quantile(ps)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a PMF from a sequence of values.

        seq: any kind of sequence
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        options: passed to the pd.Series constructor

        :return: Pmf object
        """
        series = pd.Series(seq).value_counts(sort=False)

        options["copy"] = False
        pmf = Pmf(series, **options)

        if sort:
            pmf.sort_index(inplace=True)

        if normalize:
            pmf.normalize()

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

        seq: any kind of sequence
        normalize: whether to normalize the Cdf, default True
        sort: whether to sort the Cdf by values, default True
        options: passed to the pd.Series constructor

        :return: CDF object
        """
        pmf = Pmf.from_seq(seq, normalize=False, sort=sort, **options)
        return pmf.make_cdf(normalize=normalize)

    def plot(self, **options):
        """Plot the Cdf as a line.

        :param options: passed to plt.plot

        :return:
        """
        underride(options, label=self.name, color="C4")
        plt.plot(self.qs, self.ps, **options)

    def step(self, **options):
        """Plot the Cdf as a step function.

        :param options: passed to plt.step

        :return:
        """
        underride(options, label=self.name, where="post", color="C4")
        plt.step(self.qs, self.ps, **options)

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

        :return array of probabilities
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

        :return array of quantities
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

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Cdf.

        :param normalize: Boolean, whether to normalize the Pmf

        :return: Pmf
        """
        #TODO: check for consistent behvavior of copy flag for all make_x
        normalize = kwargs.pop('normalize', False)
        ps = self.ps
        diff = np.ediff1d(ps, to_begin=ps[0])
        pmf = Pmf(diff, index=self.index.copy(), **kwargs)
        if normalize:
            pmf.normalize()
        return pmf

    def make_surv(self, **kwargs):
        """Make a Surv object from the Cdf.

        Note: You can make a Surv from an unnormalized Cdf,
        but not the other way around.

        :return: Surv object
        """
        normalize = kwargs.pop('normalize', False)
        p0 = self.ps[-1]
        surv = Surv(p0-self, **kwargs)
        if normalize:
            surv.normalize()
        return surv

    def make_hazard(self, **kwargs):
        """Make a Hazard object from the Cdf.

        :return: Hazard object
        """
        surv = self.make_surv()
        return surv.make_hazard(**kwargs)

    def choice(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `p` is provided.

        args: same as np.random.choice
        options: same as np.random.choice

        :return: NumPy array
        """
        # TODO: Make this more efficient by implementing the inverse CDF method.
        pmf = self.make_pmf()
        return pmf.choice(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `weights` is provided.

        This function returns an array containing a sample of the quantities in this Pmf,
        which is different from Series.sample, which returns a Series with a sample of
        the rows in the original Series.

        args: same as Series.sample
        options: same as Series.sample

        :return: NumPy array
        """
        # TODO: Make this more efficient by implementing the inverse CDF method.
        pmf = self.make_pmf()
        return pmf.sample(*args, **kwargs)

    def mean(self):
        """Expected value.

        :return: float
        """
        return self.make_pmf().mean()

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

        :return: float
        """
        return self.quantile(0.5)


from scipy.interpolate import interp1d


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

        seq: any kind of sequence
        normalize: whether to normalize the Surv, default True
        sort: whether to sort the Surv by values, default True
        options: passed to the pd.Series constructor

        :return: Surv object
        """
        pmf = Pmf.from_seq(seq, normalize=False, sort=sort, **options)
        cdf = pmf.make_cdf(normalize=normalize)
        return cdf.make_surv()

    def plot(self, **options):
        """Plot the Cdf as a line.

        :param options: passed to plt.plot
        :return:
        """
        underride(options, label=self.name, color="C2")
        plt.plot(self.qs, self.ps, **options)

    def step(self, **options):
        """Plot the Cdf as a step function.

        :param options: passed to plt.step
        :return:
        """
        underride(options, label=self.name, where="post", color="C2")
        plt.step(self.qs, self.ps, **options)

    def normalize(self):
        """Make the probabilities start at (modifies self).

        :return: normalizing constant
        """
        # TODO: get this under test
        total = self.ps[0]
        self /= total
        return total

    @property
    def forward(self, **kwargs):
        """Compute the forward Cdf

        :param kwargs: keyword arguments passed to interp1d

        :return array of probabilities
        """

        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(1, 0),
        )

        interp = interp1d(self.qs, self.ps, **kwargs)
        return interp

    @property
    def inverse(self, **kwargs):
        """Compute the inverse Cdf

        :param kwargs: keyword arguments passed to interp1d

        :return array of quantities
        """
        p0 = self.ps[0]
        #TODO: make this work not non-normalized Surv
        interp = self.make_cdf().inverse
        return lambda ps: interp(1 - np.asarray(ps), **kwargs)

    # calling a Cdf like a function does forward lookup
    __call__ = forward

    # quantile is the same as an inverse lookup
    quantile = inverse

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Surv.

        :return: Cdf
        """
        # TODO: error if the Surv is not normalized
        normalize = kwargs.pop('normalize', False)
        cdf = Cdf(1-self, **kwargs)
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
        """Make a Hazard object from the Surv.

        :return: Hazard object
        """
        pmf = self.make_pmf()
        at_risk = self + pmf
        haz = pmf.ps / at_risk
        return Hazard(haz, **kwargs)

    def choice(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `p` is provided.

        args: same as np.random.choice
        options: same as np.random.choice

        :return: NumPy array
        """
        # TODO: Make this more efficient by implementing the inverse CDF method.
        pmf = self.make_pmf()
        return pmf.choice(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `weights` is provided.

        This function returns an array containing a sample of the quantities in this Pmf,
        which is different from Series.sample, which returns a Series with a sample of
        the rows in the original Series.

        args: same as Series.sample
        options: same as Series.sample

        :return: NumPy array
        """
        # TODO: Make this more efficient by implementing the inverse CDF method.
        pmf = self.make_pmf()
        return pmf.sample(*args, **kwargs)

    def mean(self):
        """Expected value.

        :return: float
        """
        return self.make_pmf().mean()

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

        :return: float
        """
        return self.quantile(0.5)


class Hazard(Distribution):
    """Represents a Hazard function."""

    def copy(self, deep=True):
        """Make a copy.

        :return: new Pmf
        """
        return Hazard(self, copy=deep)

    def __getitem__(self, qs):
        """Look up qs and return ps."""
        try:
            return super().__getitem__(qs)
        except (KeyError, ValueError, IndexError):
            return 0

    # You can also call a Hazard object like a function
    __call__ = __getitem__

    def mean(self):
        """Computes expected value.

        :return: float
        """
        raise ValueError()

    def median(self):
        """Median (50th percentile).

        :return: float
        """
        raise ValueError()

    def quantile(self, ps, **kwargs):
        """Quantiles.

        Computes the inverse CDF of ps, that is,
        the values that correspond to the given probabilities.

        :return: float
        """
        raise ValueError()

    def var(self):
        """Variance of a PMF.

        :return: float
        """
        raise ValueError()

    def std(self):
        """Standard deviation of a PMF.

        :return: float
        """
        raise ValueError()

    def choice(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `p` is provided.

        args: same as np.random.choice
        kwargs: same as np.random.choice

        :return: NumPy array
        """
        raise ValueError()

    def sample(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `weights` is provided.

        This function returns an array containing a sample of the quantities,
        which is different from Series.sample, which returns a Series with a sample of
        the rows in the original Series.

        args: same as Series.sample
        options: same as Series.sample

        :return: NumPy array
        """
        raise ValueError()

    def plot(self, **options):
        """Plot the Pmf as a line.

        :param options: passed to plt.plot
        :return:
        """
        underride(options, label=self.name, color="C3")
        plt.plot(self.qs, self.ps, **options)

    def bar(self, **options):
        """Makes a bar plot.

        options: passed to plt.bar
        """
        underride(options, label=self.name)
        plt.bar(self.qs, self.ps, **options, color="C3")

    def make_surv(self, **kwargs):
        """Make a Surv from the Hazard.

        :return: Surv
        """
        normalize = kwargs.pop('normalize', False)
        ps = (1 - self).cumprod()
        surv = Surv(ps, **kwargs)
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

    @staticmethod
    def from_seq(seq, **kwargs):
        """Make a PMF from a sequence of values.

        seq: any kind of sequence
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        kwargs: passed to the pd.Series constructor

        :return: Hazard object
        """
        pmf = Pmf.from_seq(seq, **kwargs)
        return pmf.make_hazard()
