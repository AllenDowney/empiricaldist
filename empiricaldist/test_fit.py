"""Tests for empiricaldist.fit"""

import matplotlib

matplotlib.use("Agg")

import unittest
import warnings

import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist
from scipy.stats import weibull_min


from empiricaldist import (
    Cdf,
    Pmf,
    Surv,
    TailDist,
    average_error,
    cdf_bounds_to_tail,
    empirical_error_bounds,
    fit_normal,
    fit_scipy_dist,
    fit_tail_normal,
    fit_tail_t,
    fit_truncated_normal,
    fit_truncated_t,
    make_normalized_surv,
    minimize_df,
    model_error_bounds,
    plot_dist_bounds,
    plot_fit_with_area,
    plot_model_bounds,
    plot_normal_vs_lognormal,
)


class TestExports(unittest.TestCase):
    def test_public_names_importable(self):
        import empiricaldist

        for name in empiricaldist.__all__:
            self.assertTrue(hasattr(empiricaldist, name), name)


class TestFitNormal(unittest.TestCase):
    def test_fit_normal_from_series(self):
        np.random.seed(42)
        mu, sigma = 100, 15
        sample = np.random.normal(mu, sigma, size=100)
        dist = fit_normal(sample)
        self.assertAlmostEqual(dist.mean(), mu, delta=2)
        self.assertAlmostEqual(dist.std(), sigma, delta=2)

    def test_fit_normal_from_cdf(self):
        np.random.seed(42)
        mu, sigma = 100, 15
        sample = np.random.normal(mu, sigma, size=100)
        cdf = Cdf.from_seq(sample)
        dist = fit_normal(cdf)
        self.assertAlmostEqual(dist.mean(), mu, delta=2)
        self.assertAlmostEqual(dist.std(), sigma, delta=2)

    def test_fit_normal_from_pmf(self):
        np.random.seed(42)
        mu, sigma = 100, 15
        sample = np.random.normal(mu, sigma, size=100)
        pmf = Pmf.from_seq(sample)
        dist = fit_normal(pmf)
        self.assertAlmostEqual(dist.mean(), mu, delta=2)
        self.assertAlmostEqual(dist.std(), sigma, delta=2)

    def test_fit_normal_series_matches_cdf(self):
        np.random.seed(7)
        sample = np.random.normal(50, 10, size=100)
        dist_from_series = fit_normal(sample)
        dist_from_cdf = fit_normal(Cdf.from_seq(sample))
        self.assertAlmostEqual(dist_from_series.mean(), dist_from_cdf.mean())
        self.assertAlmostEqual(dist_from_series.std(), dist_from_cdf.std())

    def test_fit_normal_x0(self):
        np.random.seed(42)
        mu, sigma = 100, 15
        sample = np.random.normal(mu, sigma, size=100)
        dist = fit_normal(sample, x0=(90, 10))
        self.assertAlmostEqual(dist.mean(), mu, delta=2)
        self.assertAlmostEqual(dist.std(), sigma, delta=2)

    def test_fit_normal_matches_fit_scipy_dist(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=500)
        qs = np.linspace(70, 130, 50)

        model1 = fit_normal(sample)
        model2 = fit_scipy_dist(sample, norm)
        self.assertAlmostEqual(model1.mean(), model2.mean())
        self.assertAlmostEqual(model1.std(), model2.std())
        self.assertTrue(np.allclose(model1.cdf(qs), model2.cdf(qs)))

        cdf = Cdf.from_seq(sample)
        model3 = fit_normal(cdf)
        model4 = fit_scipy_dist(cdf, norm)
        self.assertAlmostEqual(model3.mean(), model4.mean())
        self.assertAlmostEqual(model3.std(), model4.std())
        self.assertTrue(np.allclose(model3.cdf(qs), model4.cdf(qs)))


class TestAverageError(unittest.TestCase):
    def test_average_error(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=500)
        model = fit_normal(sample)

        mae = average_error(sample, model)
        self.assertAlmostEqual(mae, 0.006281773833017781, places=10)

        cdf = Cdf.from_seq(sample)
        mae2 = average_error(cdf, model)
        self.assertAlmostEqual(mae2, 0.006281773833017781, places=10)

    def test_average_error_custom_qs(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=500)
        model = fit_normal(sample)
        qs = np.linspace(70, 130, 50)

        mae = average_error(sample, model, qs=qs)
        self.assertAlmostEqual(mae, 0.006559856561483042, places=10)


class TestFitScipyDist(unittest.TestCase):
    def test_fit_weibull_min(self):
        sample = weibull_min.rvs(
            1.85, loc=1.15, scale=22.29, size=10000, random_state=42
        )

        dist = fit_scipy_dist(sample, weibull_min)
        self.assertAlmostEqual(dist.args[0], 1.8384080705102877, places=10)
        self.assertAlmostEqual(dist.args[1], 1.2530693338263845, places=10)
        self.assertAlmostEqual(dist.args[2], 21.893020086289383, places=10)

        cdf = Cdf.from_seq(sample)
        dist2 = fit_scipy_dist(cdf, weibull_min)
        self.assertAlmostEqual(dist2.args[0], 1.8384080705102877, places=10)
        self.assertAlmostEqual(dist2.args[1], 1.2530693338263845, places=10)
        self.assertAlmostEqual(dist2.args[2], 21.893020086289383, places=10)


class TestMakeNormalizedSurv(unittest.TestCase):
    def test_make_normalized_surv_t(self):
        from scipy.stats import t as t_dist

        model = t_dist(5, loc=100, scale=10)
        qs = np.linspace(80, 120, 50)
        surv = make_normalized_surv(model, qs, q0=qs[0])
        self.assertIsInstance(surv, Surv)
        self.assertAlmostEqual(surv.ps[0], 1.0)
        expected = model.sf(qs) / model.sf(qs[0])
        np.testing.assert_allclose(surv.ps, expected)

    def test_make_normalized_surv_default_q0(self):
        from scipy.stats import norm

        model = norm(100, 15)
        qs = np.linspace(70, 130, 25)
        surv = make_normalized_surv(model, qs)
        self.assertAlmostEqual(surv.ps[0], 1.0)


class TestModelErrorBounds(unittest.TestCase):
    def test_model_error_bounds(self):
        from scipy.stats import norm

        dist = norm(100, 15)
        n = 100
        qs = np.linspace(70, 130, 50)
        low, high = model_error_bounds(dist, n, qs)
        ps = dist.cdf(qs)

        self.assertEqual(len(low.ps), len(qs))
        self.assertEqual(len(high.ps), len(qs))
        self.assertTrue(np.all(low.ps <= high.ps))
        self.assertTrue(np.all(low.ps <= ps + 1e-10))
        self.assertTrue(np.all(high.ps >= ps - 1e-10))
        self.assertIsInstance(low, Cdf)
        self.assertIsInstance(high, Cdf)

    def test_model_error_bounds_at_one(self):
        from scipy.stats import norm

        dist = norm(0, 1)
        qs = np.array([10.0])
        low, high = model_error_bounds(dist, n=100, qs=qs)
        self.assertEqual(low.ps[0], 1.0)

    def test_cdf_bounds_to_tail(self):
        from scipy.stats import norm

        dist = norm(100, 15)
        qs = np.linspace(70, 130, 50)
        low, high = model_error_bounds(dist, n=100, qs=qs)
        tail_low, tail_high = cdf_bounds_to_tail(low, high)
        self.assertIsInstance(tail_low, TailDist)
        self.assertIsInstance(tail_high, TailDist)
        self.assertTrue(np.all(tail_low.ps <= tail_high.ps + 1e-10))


class TestPlotModelBounds(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        plt.figure()
        self.plt = plt

    def tearDown(self):
        self.plt.close()

    def test_plot_model_bounds(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        model = fit_normal(sample)
        qs = np.linspace(sample.min(), sample.max(), 101)
        plot_model_bounds(model, n=len(sample), qs=qs)
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)

    def test_plot_model_bounds_custom_qs(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        model = fit_normal(sample)
        qs = np.linspace(70, 130, 25)
        plot_model_bounds(model, n=len(sample), qs=qs, color="gray", hatch="/")
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        poly = ax.collections[0]
        self.assertEqual(poly.get_facecolor()[0, 0], poly.get_edgecolor()[0, 0])

    def test_plot_model_bounds_tail_kind(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        model = fit_normal(sample)
        qs = np.linspace(70, 130, 25)
        plot_model_bounds(model, n=len(sample), qs=qs, kind="tail")
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)

    def test_plot_model_bounds_bad_kind(self):
        model = fit_normal([1, 2, 3])
        with self.assertRaises(ValueError):
            plot_model_bounds(model, n=3, qs=[1, 2, 3], kind="surv")


class TestPlotDistBounds(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        plt.figure()
        self.plt = plt

    def tearDown(self):
        self.plt.close()

    def test_plot_dist_bounds_cdf(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        cdf = Cdf.from_seq(sample)
        plot_dist_bounds(cdf, n=len(sample))
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)

    def test_plot_dist_bounds_tail(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        tail = TailDist.from_seq(sample)
        qs = np.linspace(sample.min(), sample.max(), 50)
        plot_dist_bounds(tail, n=len(sample), qs=qs, kind="tail")
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)

    def test_plot_dist_bounds_bad_kind(self):
        cdf = Cdf.from_seq([1, 2, 3])
        with self.assertRaises(ValueError):
            plot_dist_bounds(cdf, n=3, kind="surv")


class TestPlotFitWithArea(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        plt.figure()
        self.plt = plt

    def tearDown(self):
        self.plt.close()

    def test_plot_fit_with_area(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        model = fit_normal(sample)
        plot_fit_with_area(sample, model)
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 1)

    def test_plot_fit_with_area_from_cdf(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        cdf = Cdf.from_seq(sample)
        model = fit_normal(cdf)
        plot_fit_with_area(cdf, model, label="data")
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(ax.lines[0].get_label(), "data")

    def test_plot_fit_with_area_tail(self):
        np.random.seed(42)
        sample = t_dist.rvs(5, loc=100, scale=20, size=500)
        data_tail = TailDist.from_seq(sample)
        model = fit_tail_t(data_tail, df=5)
        plot_fit_with_area(data_tail, model, kind="tail")
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 1)

    def test_plot_fit_with_area_tail_from_surv(self):
        np.random.seed(42)
        sample = t_dist.rvs(5, loc=100, scale=20, size=500)
        data_tail = TailDist.from_seq(sample)
        model = fit_tail_t(data_tail, df=5)
        qs = np.linspace(data_tail.qs.min(), data_tail.qs.max(), 200)
        model_surv = make_normalized_surv(model, qs, q0=qs[0])
        plot_fit_with_area(data_tail, model_surv, kind="tail", label="data")
        ax = self.plt.gca()
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(ax.lines[0].get_label(), "data")

    def test_plot_fit_with_area_bad_kind(self):
        sample = np.random.normal(100, 15, size=100)
        model = fit_normal(sample)
        with self.assertRaises(ValueError):
            plot_fit_with_area(sample, model, kind="surv")


class TestPlotNormalVsLognormal(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        plt.figure()
        self.plt = plt

    def tearDown(self):
        self.plt.close()

    def test_plot_normal_vs_lognormal(self):
        np.random.seed(42)
        sample = np.random.lognormal(4, 0.5, size=100)
        plot_normal_vs_lognormal(sample, xlabel="value")
        fig = self.plt.gcf()
        self.assertEqual(len(fig.axes), 2)
        for ax in fig.axes:
            self.assertEqual(len(ax.collections), 1)
            self.assertEqual(len(ax.lines), 1)
        self.assertEqual(fig.axes[0].get_xlabel(), "value")
        self.assertEqual(fig.axes[1].get_xlabel(), "log10(value)")

    def test_plot_normal_vs_lognormal_from_cdf(self):
        np.random.seed(42)
        sample = np.random.lognormal(4, 0.5, size=100)
        cdf = Cdf.from_seq(sample)
        plot_normal_vs_lognormal(cdf, xlabel="value")
        fig = self.plt.gcf()
        self.assertEqual(len(fig.axes), 2)


class TestFitTruncatedNormal(unittest.TestCase):
    def test_fit_truncated_normal_recovers_parameters(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=1000)

        mu, sigma = fit_truncated_normal(sample)
        self.assertAlmostEqual(mu, 100.12981528009907, places=10)
        self.assertAlmostEqual(sigma, 14.599338603416408, places=10)

        tail = TailDist.from_seq(sample)
        mu2, sigma2 = fit_truncated_normal(tail)
        self.assertAlmostEqual(mu2, 100.12981528009907, places=10)
        self.assertAlmostEqual(sigma2, 14.599338603416408, places=10)

    def test_fit_truncated_normal_from_pmf(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=1000)
        pmf = Pmf.from_seq(sample)

        mu, sigma = fit_truncated_normal(pmf)
        self.assertAlmostEqual(mu, 100.12994277380766, places=10)
        self.assertAlmostEqual(sigma, 14.598496287184052, places=10)


class TestEmpiricalErrorBounds(unittest.TestCase):
    def test_empirical_error_bounds_from_sequence(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        qs = np.linspace(sample.min(), sample.max(), 20)
        low, high = empirical_error_bounds(sample, n=len(sample), qs=qs)
        self.assertEqual(len(low.ps), len(qs))
        self.assertEqual(len(high.ps), len(qs))
        self.assertTrue(np.all(low.ps <= high.ps))
        self.assertIsInstance(low, Cdf)

    def test_empirical_error_bounds_no_clip_warning(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        qs = np.linspace(sample.min(), sample.max(), 20)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            empirical_error_bounds(sample, n=len(sample), qs=qs)

    def test_empirical_error_bounds_warns_if_clip_large(self):
        tail = TailDist.from_seq([1, 2, 3])
        bad_cdf = Cdf([0.5, 1.2], index=[1, 2])

        class BadTail(TailDist):
            def make_cdf(self, **kwargs):
                return bad_cdf

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            empirical_error_bounds(BadTail(tail), n=10, qs=[2])
        self.assertEqual(len(caught), 1)
        self.assertIn("out of [0, 1]", str(caught[0].message))

    def test_empirical_error_bounds_from_surv(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        tail = Surv.from_seq(sample).make_tail()
        qs = np.linspace(sample.min(), sample.max(), 20)
        low, high = empirical_error_bounds(tail, n=len(sample), qs=qs)
        self.assertTrue(np.all(low.ps <= high.ps))


class TestFitTruncatedT(unittest.TestCase):
    def test_fit_truncated_t_recovers_parameters(self):
        true_df = 5
        sample = t_dist.rvs(
            true_df, loc=100, scale=10, size=1000, random_state=42
        )
        mu, sigma = fit_truncated_t(true_df, sample)
        self.assertAlmostEqual(mu, 99.95036009447568, places=10)
        self.assertAlmostEqual(sigma, 9.789873922524434, places=10)

        tail = TailDist.from_seq(sample)
        mu2, sigma2 = fit_truncated_t(true_df, tail)
        self.assertAlmostEqual(mu2, 99.95036009447568, places=10)
        self.assertAlmostEqual(sigma2, 9.789873922524434, places=10)


class TestFitTailNormal(unittest.TestCase):
    def test_fit_tail_normal_recovers_parameters(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=1000)

        dist = fit_tail_normal(sample)
        self.assertAlmostEqual(dist.kwds["loc"], 100.12981528009907, places=10)
        self.assertAlmostEqual(
            dist.kwds["scale"], 14.599338603416408, places=10
        )

        tail = TailDist.from_seq(sample)
        dist2 = fit_tail_normal(tail)
        self.assertAlmostEqual(dist2.kwds["loc"], 100.12981528009907, places=10)
        self.assertAlmostEqual(
            dist2.kwds["scale"], 14.599338603416408, places=10
        )

    def test_fit_tail_normal_returns_scipy_dist(self):
        np.random.seed(42)
        sample = np.random.normal(100, 15, size=100)
        dist = fit_tail_normal(sample)
        self.assertTrue(hasattr(dist, "sf"))
        self.assertTrue(hasattr(dist, "cdf"))
        self.assertTrue(0 < dist.sf(100) < 1)


class TestFitTailT(unittest.TestCase):
    def test_fit_tail_t_fixed_df_recovers_parameters(self):
        true_df = 5
        sample = t_dist.rvs(
            true_df, loc=100, scale=10, size=1000, random_state=42
        )

        dist = fit_tail_t(sample, df=true_df)
        self.assertEqual(dist.args[0], true_df)
        self.assertAlmostEqual(dist.kwds["loc"], 99.95036009447568, places=10)
        self.assertAlmostEqual(dist.kwds["scale"], 9.789873922524434, places=10)

        tail = TailDist.from_seq(sample)
        dist2 = fit_tail_t(tail, df=true_df)
        self.assertEqual(dist2.args[0], true_df)
        self.assertAlmostEqual(dist2.kwds["loc"], 99.95036009447568, places=10)
        self.assertAlmostEqual(
            dist2.kwds["scale"], 9.789873922524434, places=10
        )

    def test_fit_tail_t_searches_df(self):
        true_df = 5
        sample = t_dist.rvs(
            true_df, loc=100, scale=10, size=10000, random_state=42
        )

        dist = fit_tail_t(sample, df=None, bounds=(1, 1000))
        self.assertAlmostEqual(dist.args[0], 4.694855680218307, places=10)
        self.assertAlmostEqual(dist.kwds["loc"], 99.85713668957527, places=10)
        self.assertAlmostEqual(dist.kwds["scale"], 9.968598331772622, places=10)

        tail = TailDist.from_seq(sample)
        dist2 = fit_tail_t(tail, df=None, bounds=(1, 1000))
        self.assertAlmostEqual(dist2.args[0], 4.694855680218307, places=10)
        self.assertAlmostEqual(dist2.kwds["loc"], 99.85713668957527, places=10)
        self.assertAlmostEqual(
            dist2.kwds["scale"], 9.968598331772622, places=10
        )

    def test_fit_tail_t_returns_scipy_dist(self):
        sample = t_dist.rvs(5, loc=100, scale=10, size=100, random_state=42)
        dist = fit_tail_t(sample, df=5)
        self.assertTrue(hasattr(dist, "sf"))
        self.assertTrue(hasattr(dist, "cdf"))
        self.assertTrue(0 < dist.sf(100) < 1)


class TestMinimizeDf(unittest.TestCase):
    def test_minimize_df_recovers_df(self):
        true_df = 5
        sample = t_dist.rvs(
            true_df, loc=100, scale=10, size=10000, random_state=42
        )
        bounds = [(1, 1000)]

        df = minimize_df(10, sample, bounds=bounds)
        self.assertAlmostEqual(float(df[0]), 4.694855680218307, places=10)

        tail = TailDist.from_seq(sample)
        df2 = minimize_df(10, tail, bounds=bounds)
        self.assertAlmostEqual(float(df2[0]), 4.694855680218307, places=10)


if __name__ == "__main__":
    unittest.main()
