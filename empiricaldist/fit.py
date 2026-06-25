"""Fit parametric distributions to empirical data and compare them visually.

Fitting philosophy
------------------
Fitters choose a grid of **probabilities**, find the corresponding empirical
**quantiles**, and minimize **vertical error between empirical and model CDF
values** (CDF matching), not moment-matching or vanilla MLE alone. The primary
use case is graphical comparison of empirical and fitted distributions.

Loss function (``soft_l1``)
---------------------------
The default least-squares loss is ``soft_l1``, not for outlier robustness in the
usual regression sense, but because the package is primarily trying to minimize
visual discrepancy between empirical and fitted CDF/tail curves. The visual
discrepancy is closer to mean absolute vertical error — area between curves —
than to mean squared error. ``average_error`` measures that MAE directly; the
default ``loss="soft_l1"`` makes the optimizer approximate the same criterion
while staying inside SciPy's differentiable ``least_squares`` machinery.

All public functions are re-exported from ``empiricaldist`` (see ``__all__``).

Inputs
------
Bulk helpers accept a sequence, ``pd.Series``, or any ``Distribution``
(``Pmf``, ``Cdf``, ``Surv``, ``TailDist``, …) via ``_as_cdf``.

Tail helpers accept the same types via ``_as_tail`` (sequences become
``TailDist``).

Bulk fitting
------------
``fit_normal(data, ...)`` → frozen ``scipy.stats.norm``

``fit_scipy_dist(data, dist, ...)`` → frozen scipy distribution (generalizes
``fit_normal`` to Weibull, etc.)

``average_error(data, model, qs=None)`` → mean absolute CDF error on the
0–1 probability scale.

Tail fitting
------------
High-level API (returns frozen scipy distributions):

``fit_tail_normal(data, ...)`` → ``scipy.stats.norm``

``fit_tail_t(data, df=None, ...)`` → ``scipy.stats.t``; searches ``df`` when
``df`` is ``None``.

Lower-level building blocks (return ``(mu, sigma)`` or optimized ``df``):

``fit_truncated_normal``, ``fit_truncated_t``, ``minimize_df`` — fit location
and scale on a **renormalized** tail survival curve (see
``make_normalized_surv``). Used internally by the tail fitters and in notebooks
such as ``notebooks/chile.md``.

``make_normalized_surv(dist, qs, q0=None)`` → ``Surv`` with
``dist.sf(q) / dist.sf(q0)`` on a grid (for discrete model approximation and
tail-scale plots).

Surv vs TailDist (developer note)
---------------------------------
We intentionally keep ``TailDist`` for empirical data and ``Surv`` for
continuous model evaluations. This preserves the correct semantics of both
representations while allowing direct comparison, since the two notions
coincide for continuous distributions (SciPy exposes only ``dist.sf``).

Error bounds
------------
``model_error_bounds(dist, n, qs, ...)`` — if the model were true, where could
a sample CDF fall? (binomial bands)

``empirical_error_bounds(data, n, qs, ...)`` — given this empirical
distribution, where could the true CDF fall in another sample?

``cdf_bounds_to_tail(low_cdf, high_cdf)`` — pointwise complement for tail-scale
plots.

Plotting
--------
``plot_fit_with_area(data, model, kind="cdf"|"tail", ...)`` — empirical curve
with shaded vertical gap to the model (CDF or survival scale).

``plot_model_bounds(model, n, qs, kind=..., ...)`` — confidence band only;
caller draws the model curve.

``plot_dist_bounds(dist, n, qs=None, kind=..., ...)`` — band for empirical or
discrete model ``Distribution``; caller draws the curve.

``plot_normal_vs_lognormal(data, ...)`` — side-by-side normal vs lognormal
comparison using area plots.

Examples
--------
See ``notebooks/test_fit.md`` (bulk fit, bounds, tail fitting) and
``notebooks/chile.md``, ``notebooks/cricket.md``, ``notebooks/president_normality.md``.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import binom, norm, t as t_dist

from .empiricaldist import Cdf, Distribution, Pmf, Surv, TailDist, underride


def _as_cdf(data):
    """Convert a sequence, Series, or Distribution to a Cdf."""
    if isinstance(data, Cdf):
        return data
    if isinstance(data, Distribution):
        return data.make_cdf()
    return Cdf.from_seq(data)


def _as_tail(data):
    """Convert a sequence, Series, or Distribution to a TailDist."""
    if isinstance(data, TailDist):
        return data
    if isinstance(data, Distribution):
        return data.make_tail()
    return TailDist.from_seq(data)


def fit_normal(data, ps=None, x0=None, loss="soft_l1"):
    """Fit a normal distribution by minimizing vertical CDF error.

    Chooses a grid of probabilities, finds the corresponding empirical
    quantiles, and minimizes vertical error between empirical and model CDF
    values.

    Args:
        data: sequence, ``pd.Series``, or an empiricaldist ``Distribution``
        ps: probabilities to use for fitting; default 99 points from 0.01 to 0.99
        x0: optional initial guess (mu, sigma); defaults to sample mean and std
        loss: loss function passed to ``scipy.optimize.least_squares``; default
            ``soft_l1`` approximates absolute vertical CDF error (the visual
            criterion used by ``average_error`` and the area plots)

    Returns:
        scipy.stats.norm object
    """
    return fit_scipy_dist(data, norm, ps=ps, x0=x0, loss=loss)


def _sample_for_scipy_fit(data, cdf):
    """Sample array for scipy ``dist.fit`` initial guesses."""
    if isinstance(data, Distribution):
        return np.asarray(cdf.qs)[np.isfinite(cdf.qs)]
    sample = np.asarray(data, dtype=float)
    return sample[np.isfinite(sample)]


def fit_scipy_dist(data, dist, ps=None, x0=None, loss="soft_l1"):
    """Fit a scipy.stats distribution by minimizing vertical CDF error.

    Chooses a grid of probabilities, finds the corresponding empirical
    quantiles, and minimizes vertical error between empirical and model CDF
    values.

    Args:
        data: sequence, ``pd.Series``, or an empiricaldist ``Distribution``
        dist: scipy.stats continuous distribution (e.g. ``norm``, ``weibull_min``)
        ps: probabilities to match; default 99 points from 0.01 to 0.99
        x0: initial parameter guess; default scipy MLE via ``dist.fit()``
        loss: loss function passed to ``scipy.optimize.least_squares``; default
            ``soft_l1`` approximates absolute vertical CDF error (the visual
            criterion used by ``average_error`` and the area plots)

    Returns:
        frozen scipy distribution
    """
    cdf = _as_cdf(data)
    if ps is None:
        ps = np.linspace(0.01, 0.99, 99)
    qs = cdf.quantile(ps)

    def error_func(params):
        return ps - dist.cdf(qs, *params)

    if x0 is None:
        sample = _sample_for_scipy_fit(data, cdf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x0 = dist.fit(sample)
    res = least_squares(error_func, x0=x0, loss=loss)
    if not res.success:
        raise RuntimeError(f"fit_scipy_dist failed: {res.message}")
    return dist(*res.x)


def average_error(data, model, qs=None):
    """Mean absolute vertical CDF error between data and a fitted model.

    Args:
        data: sequence, ``pd.Series``, or an empiricaldist ``Distribution``
        model: fitted distribution with a ``cdf`` method
        qs: quantities at which to compare; default grid spans the 1st through
            99th empirical quantiles (probabilities 0.01–0.99)

    Returns:
        Mean of ``|F_empirical(q) - F_model(q)|`` over ``qs`` (probability
        scale, 0–1).
    """
    cdf = _as_cdf(data)
    if qs is None:
        ps = np.linspace(0.01, 0.99, 99)
        qs = cdf.quantile(ps)
    else:
        qs = np.asarray(qs)
    return np.mean(np.abs(cdf(qs) - model.cdf(qs)))


def _binom_cdf_bounds(ps, n, qs, con_level=0.95):
    """Binomial confidence bounds on CDF probabilities."""
    p_low = (1 - con_level) / 2
    p_high = 1 - p_low

    low = binom.ppf(p_low, n, ps) / n
    high = binom.ppf(p_high, n, ps) / n
    at_one = ps >= 1
    low[at_one] = 1
    high[at_one] = 1
    return Cdf(low, index=qs), Cdf(high, index=qs)


def cdf_bounds_to_tail(low_cdf, high_cdf):
    """Convert CDF error bounds to tail scale.

    At each quantity ``q``, applies the pointwise complement with swapped
    bounds: ``tail_low(q) = 1 - high_cdf(q)`` and
    ``tail_high(q) = 1 - low_cdf(q)``.

    This is not the same as ``Cdf.make_tail()``, which converts an
    empirical CDF to ``P(X >= x)`` for the *same* discrete distribution
    (``tail(q_i) = 1 - cdf(q_{i-1})``, not ``1 - cdf(q_i)``). Binomial
    error bounds are pointwise limits on ``P(X <= q)``; complementing
    them is the right operation for tail-scale plots.
    """
    tail_low = TailDist(1 - high_cdf.ps, index=high_cdf.index)
    tail_high = TailDist(1 - low_cdf.ps, index=low_cdf.index)
    return tail_low, tail_high


def model_error_bounds(dist, n, qs, con_level=0.95):
    """Find confidence bounds on a model CDF.

    Uses the binomial sampling distribution: if the true probability at
    each quantity is ``p``, the observed fraction in a sample of size ``n``
    follows Binomial(n, p).

    Args:
        dist: scipy distribution object (or any object with a ``cdf`` method)
        n: sample size
        qs: quantities at which to evaluate bounds
        con_level: confidence level

    Returns:
        tuple of ``Cdf`` objects (low, high) with values in [0, 1]
    """
    return _binom_cdf_bounds(dist.cdf(qs), n, qs, con_level)


def empirical_error_bounds(data, n, qs, con_level=0.95):
    """Binomial confidence bounds on an empirical CDF.

    Args:
        data: sequence, ``pd.Series``, or ``Distribution`` (converted to ``Cdf``)
        n: sample size
        qs: quantities at which to evaluate bounds
        con_level: confidence level

    Returns:
        tuple of ``Cdf`` objects (low, high) with values in [0, 1]
    """
    cdf = _as_cdf(data)
    # Interpolated CDF lookup can overshoot [0, 1] slightly at grid endpoints;
    # clip so binom.ppf gets valid probabilities. Warn if clipping is non-trivial.
    cdf_raw = cdf(qs)
    cdf_ps = np.clip(cdf_raw, 0, 1)
    clip_err = np.max(np.abs(cdf_raw - cdf_ps))
    if clip_err > 1e-9:
        warnings.warn(
            f"CDF probabilities out of [0, 1] by up to {clip_err:g}; "
            "check data or evaluation grid",
            stacklevel=2,
        )

    return _binom_cdf_bounds(cdf_ps, n, qs, con_level)


def _check_kind(kind):
    """Validate plot kind for bounds helpers."""
    if kind not in ("cdf", "tail"):
        raise ValueError(f"kind must be 'cdf' or 'tail', got {kind!r}")


def _bounds_for_kind(low, high, kind):
    """Convert CDF bounds to tail scale when ``kind`` is ``'tail'``."""
    _check_kind(kind)
    if kind == "tail":
        return cdf_bounds_to_tail(low, high)
    return low, high


def plot_model_bounds(model, n, qs, kind="cdf", con_level=0.95, **options):
    """Plot binomial confidence bounds for a fitted model.

    Uses ``model_error_bounds``: if the model were true, where could a
    sample CDF fall? Bounds are computed on the CDF scale; when
    ``kind="tail"`` they are converted with ``cdf_bounds_to_tail``.

    Draws the band only — caller plots the model curve separately.
    For truncated models, evaluate and normalize the model at the desired
    ``qs``, build a discrete ``Distribution``, and use ``plot_dist_bounds``
    instead (see ``notebooks/test_fit.md``).

    Args:
        model: fitted scipy distribution with ``cdf`` and ``sf`` methods
        n: sample size for binomial error bounds
        qs: quantities at which to evaluate
        kind: ``"cdf"`` or ``"tail"``
        con_level: confidence level passed to ``model_error_bounds``
        **options: keyword arguments passed to ``plt.fill_between``
    """
    low, high = model_error_bounds(model, n, qs, con_level=con_level)
    low, high = _bounds_for_kind(low, high, kind)
    underride(options, linewidth=0, alpha=0.3)
    plt.fill_between(low.qs, low.ps, high.ps, **options)


def plot_dist_bounds(dist, n, qs=None, kind="cdf", con_level=0.95, **options):
    """Plot binomial confidence bounds for a ``Distribution``.

    Uses ``empirical_error_bounds``: given this distribution, where could
    the true CDF fall in another sample of size ``n``? Works for empirical
    data (``Cdf``, ``TailDist``, etc.) and for discrete approximations of
    a mathematical model evaluated on a grid (see ``notebooks/test_fit.md``,
    "Tail fitting").

    Draws the band only — caller plots the distribution curve separately.

    Bounds are computed on the CDF scale; when ``kind="tail"`` they are
    converted with ``cdf_bounds_to_tail``.

    Args:
        dist: ``Distribution`` (empirical or discrete model approximation)
        n: sample size for binomial error bounds
        qs: quantities at which to evaluate; defaults to ``dist.qs``
        kind: ``"cdf"`` or ``"tail"``
        con_level: confidence level passed to ``empirical_error_bounds``
        **options: keyword arguments passed to ``plt.fill_between``
    """
    if qs is None:
        qs = np.asarray(dist.qs)
    low, high = empirical_error_bounds(dist, n, qs, con_level=con_level)
    low, high = _bounds_for_kind(low, high, kind)
    underride(options, linewidth=0, alpha=0.3)
    plt.fill_between(low.qs, low.ps, high.ps, **options)


def plot_fit_with_area(data, model, kind="cdf", **options):
    """Plot empirical distribution with shaded area between data and model.

    The gray band shows the vertical gap between the model and the empirical
    curve at each quantity in the data.

    Args:
        data: sequence, ``pd.Series``, or an empiricaldist ``Distribution``
        model: fitted distribution with ``cdf`` / ``sf`` methods, or a
            ``Distribution`` on the same grid (e.g. from ``make_normalized_surv``)
        kind: ``"cdf"`` for bulk CDF comparison, ``"tail"`` for survival /
            ``P(X >= x)`` comparison
        **options: keyword arguments passed to the empirical ``Distribution.plot``
    """
    _check_kind(kind)
    if kind == "cdf":
        dist = _as_cdf(data)
        qs = dist.qs
        model_ps = model.cdf(qs)
    else:
        dist = _as_tail(data)
        qs = dist.qs
        if hasattr(model, "sf"):
            model_ps = model.sf(qs)
        elif isinstance(model, Distribution):
            model_ps = model(qs)
        else:
            raise TypeError(
                "model must have an sf method or be a Distribution when kind='tail'"
            )
    plt.fill_between(qs, model_ps, dist.values, color="gray", alpha=0.4)
    underride(options, lw=1)
    dist.plot(**options)


def plot_normal_vs_lognormal(data, xlabel="", figsize=(9, 4), **options):
    """Side-by-side normal and lognormal model fits using area plots.

    Left panel: fit a normal to the data. Right panel: fit a normal to
    log10(data), which corresponds to a lognormal model on the original scale.

    Args:
        data: sequence of positive values, ``pd.Series``, or ``Distribution``
        xlabel: label for the original-scale x axis; log panel uses log10(xlabel)
        figsize: figure size passed to ``plt.figure``
        **options: keyword arguments passed to ``plot_fit_with_area``
    """
    cdf = _as_cdf(data)
    log_cdf = cdf.transform(np.log10)

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    normal_model = fit_normal(cdf)
    plot_fit_with_area(cdf, normal_model, **options)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title("Normal model")

    plt.subplot(1, 2, 2)
    lognormal_model = fit_normal(log_cdf)
    plot_fit_with_area(log_cdf, lognormal_model, **options)
    log_xlabel = f"log10({xlabel})" if xlabel else "log10(x)"
    plt.xlabel(log_xlabel)
    plt.ylabel("CDF")
    plt.title("Lognormal model")

    plt.tight_layout()


# ---------------------------------------------------------------------------
# Tail fitters (Task 6)
# ---------------------------------------------------------------------------


def fit_tail_normal(data, ps=None, x0=None, loss="soft_l1"):
    """Fit a normal distribution to tail data by minimizing survival error.

    Chooses a grid of tail probabilities, finds the corresponding empirical
    quantiles, and minimizes vertical error between empirical and model
    survival values.

    Args:
        data: sequence, ``pd.Series``, or ``Distribution`` (converted to ``TailDist``)
        ps: tail probabilities to match; default 20 points from 0.10 to 0.99
        x0: optional initial guess (mu, sigma); defaults to PMF mean and std
        loss: loss function passed to ``scipy.optimize.least_squares``; default
            ``soft_l1`` approximates absolute vertical CDF/tail error (the visual
            criterion used by the plotting utilities)

    Returns:
        scipy.stats.norm object
    """
    tail = _as_tail(data)
    q0 = tail.qs.min()
    mu, sigma = fit_truncated_normal(tail, ps=ps, x0=x0, q0=q0, loss=loss)
    return norm(loc=mu, scale=sigma)


def fit_tail_t(
    data,
    df=None,
    ps=None,
    x0=None,
    df0=10,
    bounds=(1, 1000),
    loss="soft_l1",
):
    """Fit a t distribution to tail data by minimizing survival error.

    Chooses a grid of tail probabilities, finds the corresponding empirical
    quantiles, and minimizes vertical error between empirical and model
    survival values.

    Args:
        data: sequence, ``pd.Series``, or ``Distribution`` (converted to ``TailDist``)
        df: degrees of freedom; if None, search over ``bounds``
        ps: tail probabilities to match when fitting loc and scale
        x0: optional initial guess (mu, sigma)
        df0: initial guess for df when ``df`` is None
        bounds: (low, high) bounds on df when searching
        loss: loss function passed to ``scipy.optimize.least_squares``; default
            ``soft_l1`` approximates absolute vertical CDF/tail error (the visual
            criterion used by the plotting utilities)

    Returns:
        scipy.stats.t object
    """
    tail = _as_tail(data)
    q0 = tail.qs.min()
    if df is None:
        df = float(minimize_df(df0, tail, q0=q0, ps=ps, loss=loss, bounds=[bounds])[0])
    mu, sigma = fit_truncated_t(df, tail, q0=q0, ps=ps, x0=x0, loss=loss)
    return t_dist(df, loc=mu, scale=sigma)


# ---------------------------------------------------------------------------
# Renormalized tail survival functions
# ---------------------------------------------------------------------------


def make_normalized_surv(dist, qs, q0=None):
    """Normalized survival function for a scipy distribution on a grid.

    Evaluates the SciPy survival function ``dist.sf(q)`` and renormalizes by
    ``dist.sf(q0)``. Returns ``Surv(ps / ps0, qs)`` where ``ps = dist.sf(qs)``
    and ``ps0 = dist.sf(q0)``.

    The result is returned as a ``Surv`` because it is derived from a
    continuous survival function. Although this curve is compared directly with
    empirical ``TailDist`` objects during tail fitting and plotting, the
    distinction between ``P(X > x)`` and ``P(X >= x)`` vanishes for continuous
    distributions.

    Args:
        dist: scipy frozen distribution with an ``sf`` method
        qs: quantities at which to evaluate
        q0: renormalization point; default is ``qs[0]``

    Returns:
        ``Surv`` with normalized tail probabilities on ``qs``
    """
    qs = np.asarray(qs)
    if q0 is None:
        q0 = qs[0]
    ps = dist.sf(qs) / dist.sf(q0)
    return Surv(ps, index=qs)


def fit_truncated_normal(data, ps=None, x0=None, q0=None, loss="soft_l1"):
    """Fit mu and sigma for a truncated normal tail distribution.

    Args:
        data: sequence, ``pd.Series``, or ``Distribution`` (converted to ``TailDist``)
        ps: tail probabilities to match; default 20 points from 0.10 to 0.99
        x0: optional initial guess (mu, sigma)
        q0: renormalization point; default is the minimum quantity in ``data``
        loss: loss function passed to ``scipy.optimize.least_squares``

    Returns:
        tuple (mu, sigma)
    """
    tail = _as_tail(data)
    if ps is None:
        ps = np.linspace(0.1, 0.99, 20)
    if q0 is None:
        q0 = tail.qs.min()
    qs = tail.inverse(ps)

    def error_func(params):
        mu, sigma = params
        model = norm(loc=mu, scale=sigma)
        surv = make_normalized_surv(model, qs, q0)
        return ps - surv.ps

    if x0 is None:
        pmf = tail.make_pmf()
        pmf.normalize()
        x0 = pmf.mean(), pmf.std()
    res = least_squares(error_func, x0=x0, loss=loss, xtol=1e-5)
    if not res.success:
        raise RuntimeError(f"fit_truncated_normal failed: {res.message}")
    return res.x


def fit_truncated_t(df, data, q0=None, ps=None, x0=None, loss="soft_l1"):
    """Fit mu and sigma for a truncated t tail with fixed df.

    Args:
        df: degrees of freedom
        data: sequence, ``pd.Series``, or ``Distribution`` (converted to ``TailDist``)
        q0: renormalization point; default is the minimum quantity in ``data``
        ps: tail probabilities to match; default 30 points from 0.10 to 0.90
        x0: optional initial guess (mu, sigma)
        loss: loss function passed to ``scipy.optimize.least_squares``

    Returns:
        tuple (mu, sigma)
    """
    tail = _as_tail(data)
    if ps is None:
        ps = np.linspace(0.10, 0.90, 30)
    if q0 is None:
        q0 = tail.qs.min()
    qs = tail.inverse(ps)

    def error_func(params):
        mu, sigma = params
        model = t_dist(df, loc=mu, scale=sigma)
        surv = make_normalized_surv(model, qs, q0)
        return ps - surv.ps

    if x0 is None:
        pmf = tail.make_pmf()
        pmf.normalize()
        x0 = pmf.mean(), pmf.std()
    res = least_squares(error_func, x0=x0, loss=loss, xtol=1e-5)
    if not res.success:
        raise RuntimeError(f"fit_truncated_t failed: {res.message}")
    return res.x


def minimize_df(df0, data, q0=None, ps=None, loss="soft_l1", **min_options):
    """Choose df to minimize log-scale tail error for a truncated t model.

    For each candidate df, ``fit_truncated_t`` is run to find mu and sigma.
    Default probability grid follows ``examples/chile.md``.

    Args:
        df0: initial guess for df
        data: sequence, ``pd.Series``, or ``Distribution`` (converted to ``TailDist``)
        q0: renormalization point; default is the minimum quantity in ``data``
        ps: tail probabilities for the log-scale error criterion; default
            log-spaced grid. Not passed to ``fit_truncated_t`` — the inner
            fitter uses its own probability grid.
        loss: loss function passed to ``scipy.optimize.least_squares`` in
            ``fit_truncated_t``
        min_options: keyword arguments passed to ``scipy.optimize.minimize``;
            default ``method="Powell"``, ``bounds=[(1, 1e6)]``, ``options={"xtol": 1e-6}``

    Returns:
        array with optimized df
    """
    tail = _as_tail(data)
    if q0 is None:
        q0 = tail.qs.min()

    if ps is None:
        t = 0.1, tail.ps[-2]
        log_low, log_high = np.log10(t)
        ps = np.logspace(log_low, log_high, 30, endpoint=False)

    # Empirical quantiles at the probability grid, and tail probabilities there
    qs = tail.inverse(ps)
    tail_ps = tail(qs)

    def error_func_tail(params):
        (df,) = params
        mu, sigma = fit_truncated_t(df, tail, q0=q0, loss=loss)
        model = t_dist(df, loc=mu, scale=sigma)
        surv = make_normalized_surv(model, qs, q0)
        errors = np.log10(ps) - np.log10(surv.ps)
        return np.sum(errors**2)

    underride(min_options, method="Powell", bounds=[(1, 1e6)], options={"xtol": 1e-6})

    res = minimize(error_func_tail, x0=(df0,), **min_options)
    if not res.success:
        raise RuntimeError(f"minimize_df failed: {res.message}")
    return res.x
