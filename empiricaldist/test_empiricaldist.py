"""Test code for empiricaldist

Copyright 2019 Allen B. Downey
BSD 3-clause license: https://opensource.org/licenses/BSD-3-Clause
"""

import unittest
from collections import Counter

import numpy as np
import pandas as pd

from empiricaldist import Hist, Cdf, Hazard, Pmf, Surv


class Test(unittest.TestCase):
    def testHist(self):
        t = list("allen")
        hist = Hist.from_seq(t, sort=False)

        self.assertEqual(len(hist), 4)
        self.assertEqual(hist["l"], 2)

        pmf = Pmf(hist)
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf["l"], 2)

    def testPmf(self):
        t = list("allen")
        pmf = Pmf.from_seq(t)

        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf["l"], 0.4)

        pmf = Pmf(pmf)
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf["l"], 0.4)

        pmf = Pmf(Counter(t))
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf["l"], 2)

        pmf2 = pmf.copy()
        self.assertEqual(len(pmf2), 4)
        self.assertEqual(pmf2["l"], 2)

        # test choice
        np.random.seed(42)
        pmf.normalize()
        xs = pmf.choice(7, replace=True)
        self.assertListEqual(xs.tolist(), ["l", "n", "e", "l", "a", "a", "a"])

        # test a Pmf with an explicit 0
        t = [2, 1, 2, 3, 5]
        pmf = Pmf.from_seq(t, normalize=False)
        pmf[0] = 0
        pmf.sort_index(inplace=True)
        self.assertListEqual(list(pmf), [0, 1, 2, 1, 1])

        self.assertEqual(pmf(3), 1)
        self.assertEqual(pmf(4), 0)
        self.assertEqual(pmf("a"), 0)

        xs = [0, 1, 2, 3, 4, 5, 6]
        res = pmf(xs)
        self.assertListEqual(list(res), [0, 1, 2, 1, 0, 1, 0])

    def testSort(self):
        t = list("allen")
        pmf = Pmf.from_seq(t, sort=False)
        pmf.sort_index(inplace=True)
        self.assertEqual(pmf.qs[0], "a")
        self.assertEqual(pmf.qs[-1], "n")

        cdf = pmf.make_cdf()
        self.assertEqual(cdf.qs[0], "a")
        self.assertEqual(cdf.qs[-1], "n")

        # for a while it would sort anyway;
        # most recently, it doesn't.
        # So this should not be a test!
        pmf = Pmf.from_seq([3, 6, 1, 7, 2], sort=False)
        # self.assertEqual(pmf.qs[0], 1)

        # check that we can deal with an unsorted Pmf
        qs = [3, 1, 2, 4]
        ps = [0.1, 0.2, 0.3, 0.4]
        pmf = Pmf(ps, qs)
        self.assertEqual(pmf.mean(), 2.7)

        # when you make a Cdf, it gets sorted, because otherwise
        # it's not a sensible Cdf
        cdf = pmf.make_cdf()
        self.assertEqual(cdf.quantile(0.3), 2)
        self.assertEqual(cdf.quantile(0.6), 3)

    def testStats(self):
        pmf = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(pmf.mean(), 3.5)
        self.assertAlmostEqual(pmf.var(), 2.91666666)
        self.assertAlmostEqual(pmf.std(), 1.70782512)
        self.assertAlmostEqual(pmf.median(), 3)
        self.assertAlmostEqual(pmf.mode(), 1)
        self.assertAlmostEqual(pmf.quantile(0.8), 5)

        cdf = pmf.make_cdf()
        self.assertAlmostEqual(cdf.mean(), 3.5)
        self.assertAlmostEqual(cdf.var(), 2.91666666)
        self.assertAlmostEqual(cdf.std(), 1.70782512)
        self.assertAlmostEqual(cdf.median(), 3)
        self.assertAlmostEqual(cdf.quantile(0.8), 5)

        surv = pmf.make_surv()
        self.assertAlmostEqual(surv.mean(), 3.5)
        self.assertAlmostEqual(surv.var(), 2.91666666)
        self.assertAlmostEqual(surv.std(), 1.70782512)
        self.assertAlmostEqual(surv.median(), 3)
        self.assertAlmostEqual(surv.quantile(0.8), 5)

        haz = pmf.make_hazard()
        self.assertAlmostEqual(haz.mean(), 3.5)
        self.assertAlmostEqual(haz.var(), 2.91666666)
        self.assertAlmostEqual(haz.std(), 1.70782512)
        self.assertAlmostEqual(haz.median(), 3)
        self.assertAlmostEqual(haz.quantile(0.8), 5)

        haz = cdf.make_hazard()
        self.assertAlmostEqual(haz.mean(), 3.5)
        self.assertAlmostEqual(haz.var(), 2.91666666)
        self.assertAlmostEqual(haz.std(), 1.70782512)
        self.assertAlmostEqual(haz.median(), 3)
        self.assertAlmostEqual(haz.quantile(0.8), 5)

        pmf = Pmf.from_seq([1, 2, 3, 3, 4, 5, 6])
        self.assertAlmostEqual(pmf.mode(), 3)
        cdf = pmf.make_cdf()
        self.assertAlmostEqual(cdf.mode(), 3)

    def testSample(self):
        pmf = Pmf.from_seq([1, 2, 2, 4, 5])
        expected = [2, 2, 1, 1, 4, 4, 4, 2, 1, 2]

        np.random.seed(17)
        a = pmf.sample(10)
        self.assertTrue(np.all((a == expected)))

        np.random.seed(17)
        a = pmf.make_cdf().sample(10)
        self.assertTrue(np.all((a == expected)))

        np.random.seed(17)
        a = pmf.make_surv().sample(10)
        self.assertTrue(np.all((a == expected)))

        np.random.seed(17)
        a = pmf.make_hazard().sample(10)
        self.assertTrue(np.all((a == expected)))

    def testChoice(self):
        pmf = Pmf.from_seq([1, 2, 2, 4, 5])
        expected = [2, 2, 1, 1, 4, 4, 4, 2, 1, 2]

        np.random.seed(17)
        a = pmf.choice(10)
        self.assertTrue(np.all((a == expected)))

        np.random.seed(17)
        a = pmf.make_cdf().choice(10)
        self.assertTrue(np.all((a == expected)))

        np.random.seed(17)
        a = pmf.make_surv().choice(10)
        self.assertTrue(np.all((a == expected)))

        np.random.seed(17)
        a = pmf.make_hazard().choice(10)
        self.assertTrue(np.all((a == expected)))

    def testHead(self):
        pmf1 = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        h = pmf1.head()
        self.assertEqual(type(h), type(pmf1))

        cdf1 = pmf1.make_cdf()
        h = cdf1.head()
        self.assertEqual(type(h), type(cdf1))

    def testTransform(self):
        pmf1 = Pmf.from_seq([1, 2, 3, 4, 5])
        pmf2 = pmf1.transform(lambda x: x**2)
        self.assertAlmostEqual(pmf2.mean(), 11.0)

    def testAdd(self):
        pmf1 = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        pmf2 = Pmf.from_seq([1, 2, 3, 4])

        total = pmf1 + pmf2
        total.normalize()
        self.assertAlmostEqual(total.mean(), 3)

        total = pmf1.add(pmf2)
        total.normalize()
        self.assertAlmostEqual(total.mean(), 3)

    def testMul(self):
        pmf1 = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        pmf2 = Pmf.from_seq([1, 2, 3, 4])
        pmf3 = 0.5 * pmf1 + 0.5 * pmf2
        self.assertAlmostEqual(pmf3.mean(), 3.0)

    def testAddDist(self):
        pmf = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        series = pd.Series(1 / 12, pmf.index)

        total = pmf + series
        self.assertAlmostEqual(total[1], 3 / 12)

        total = series + pmf
        self.assertAlmostEqual(total[1], 3 / 12)

        pmf1 = pmf.add_dist(1)
        self.assertAlmostEqual(pmf1.mean(), 4.5)

        pmf2 = pmf.add_dist(pmf)
        self.assertAlmostEqual(pmf2.mean(), 7.0)

        cdf = pmf.make_cdf()
        cdf2 = cdf.add_dist(cdf)
        self.assertAlmostEqual(cdf2.mean(), 7.0)

        surv = pmf.make_surv()
        surv2 = surv.add_dist(surv)
        self.assertAlmostEqual(surv2.mean(), 7.0)

        haz = pmf.make_hazard()
        haz2 = haz.add_dist(haz)
        self.assertAlmostEqual(haz2.mean(), 7.0)

    def testSubDist(self):
        pmf = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        series = pd.Series(1 / 12, pmf.index)

        diff = pmf - series
        self.assertAlmostEqual(diff[1], 1 / 12)

        diff = series - pmf
        self.assertAlmostEqual(diff[1], -1 / 12)

        pmf3 = pmf.sub_dist(1)
        self.assertAlmostEqual(pmf3.mean(), 2.5)

        pmf4 = pmf.sub_dist(pmf)
        self.assertAlmostEqual(pmf4.mean(), 0)

        cdf = pmf.make_cdf()
        cdf2 = cdf.sub_dist(cdf)
        self.assertAlmostEqual(cdf2.mean(), 0)

        surv = pmf.make_surv()
        surv2 = surv.sub_dist(surv)
        self.assertAlmostEqual(surv2.mean(), 0)

        haz = pmf.make_hazard()
        haz2 = haz.sub_dist(haz)
        self.assertAlmostEqual(haz2.mean(), 0)

    def testMulDist(self):
        pmf = Pmf.from_seq([1, 2, 3, 4])
        series = pd.Series(2, pmf.index)

        prod = pmf * series
        self.assertAlmostEqual(prod[1], 0.5)

        prod = series * pmf
        self.assertAlmostEqual(prod[1], 0.5)

        pmf3 = pmf.mul_dist(2)
        self.assertAlmostEqual(pmf3.mean(), 5)

        pmf4 = pmf.mul_dist(pmf)
        self.assertAlmostEqual(pmf4.mean(), 6.25)

        cdf = pmf.make_cdf()
        cdf2 = cdf.mul_dist(cdf)
        self.assertAlmostEqual(cdf2.mean(), 6.25)

        surv = pmf.make_surv()
        surv2 = surv.mul_dist(surv)
        self.assertAlmostEqual(surv2.mean(), 6.25)

        haz = pmf.make_hazard()
        haz2 = haz.mul_dist(haz)
        self.assertAlmostEqual(haz2.mean(), 6.25)

    def testDivDist(self):
        pmf = Pmf.from_seq([1, 2, 3, 4])
        series = pd.Series(2, pmf.index)

        q = pmf / series
        self.assertAlmostEqual(q[1], 0.125)

        q = series / pmf
        self.assertAlmostEqual(q[1], 8)

        pmf3 = pmf.div_dist(2)
        self.assertAlmostEqual(pmf3.mean(), 1.25)

        pmf4 = pmf.div_dist(pmf)
        self.assertAlmostEqual(pmf4.mean(), 1.3020833333)

        cdf = pmf.make_cdf()
        cdf2 = cdf.div_dist(cdf)
        self.assertAlmostEqual(cdf2.mean(), 1.3020833333)

        surv = pmf.make_surv()
        surv2 = surv.div_dist(surv)
        self.assertAlmostEqual(surv2.mean(), 1.3020833333)

        haz = pmf.make_hazard()
        haz2 = haz.div_dist(haz)
        self.assertAlmostEqual(haz2.mean(), 1.3020833333)

    def test_joint(self):
        pmf1 = Pmf.from_seq([1, 2, 2])
        pmf2 = Pmf.from_seq([1, 2, 3])

        joint = Pmf.make_joint(pmf1, pmf2)

        mar1 = joint.marginal(0)
        mar2 = joint.marginal(1)
        self.assertAlmostEqual(mar1.mean(), pmf1.mean())
        self.assertAlmostEqual(mar2.mean(), pmf2.mean())

        cond1 = joint.conditional(0, 1)
        cond2 = joint.conditional(1, 1)
        self.assertAlmostEqual(cond1.mean(), pmf2.mean())
        self.assertAlmostEqual(cond2.mean(), pmf1.mean())

    def testComparison(self):
        pmf1 = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        pmf2 = Pmf.from_seq([1, 2, 3, 4])

        self.assertAlmostEqual(pmf1.eq_dist(3), 1 / 6)
        self.assertAlmostEqual(pmf1.ne_dist(3), 5 / 6)
        self.assertAlmostEqual(pmf1.gt_dist(3), 3 / 6)
        self.assertAlmostEqual(pmf1.ge_dist(3), 4 / 6)
        self.assertAlmostEqual(pmf1.lt_dist(3), 2 / 6)
        self.assertAlmostEqual(pmf1.le_dist(3), 3 / 6)

        self.assertAlmostEqual(pmf1.eq_dist(pmf2), 1 / 6)
        self.assertAlmostEqual(pmf1.ne_dist(pmf2), 5 / 6)
        self.assertAlmostEqual(pmf1.gt_dist(pmf2), 0.5833333)
        self.assertAlmostEqual(pmf1.ge_dist(pmf2), 3 / 4)
        self.assertAlmostEqual(pmf1.lt_dist(pmf2), 1 / 4)
        self.assertAlmostEqual(pmf1.le_dist(pmf2), 0.41666666)

        self.assertAlmostEqual(pmf1.prob_eq(3), 1 / 6)
        self.assertAlmostEqual(pmf1.prob_ne(3), 5 / 6)
        self.assertAlmostEqual(pmf1.prob_gt(3), 3 / 6)
        self.assertAlmostEqual(pmf1.prob_ge(3), 4 / 6)
        self.assertAlmostEqual(pmf1.prob_lt(3), 2 / 6)
        self.assertAlmostEqual(pmf1.prob_le(3), 3 / 6)

        self.assertAlmostEqual(pmf1.prob_eq(pmf2), 1 / 6)
        self.assertAlmostEqual(pmf1.prob_ne(pmf2), 5 / 6)
        self.assertAlmostEqual(pmf1.prob_gt(pmf2), 0.5833333)
        self.assertAlmostEqual(pmf1.prob_ge(pmf2), 3 / 4)
        self.assertAlmostEqual(pmf1.prob_lt(pmf2), 1 / 4)
        self.assertAlmostEqual(pmf1.prob_le(pmf2), 0.41666666)

    def testPmfComparison(self):
        d4 = Pmf.from_seq(range(1, 5))
        self.assertEqual(d4.gt_dist(2), 0.5)
        self.assertEqual(d4.gt_dist(d4), 0.375)

        self.assertEqual(d4.lt_dist(2), 0.25)
        self.assertEqual(d4.lt_dist(d4), 0.375)

        self.assertEqual(d4.ge_dist(2), 0.75)
        self.assertEqual(d4.ge_dist(d4), 0.625)

        self.assertEqual(d4.le_dist(2), 0.5)
        self.assertEqual(d4.le_dist(d4), 0.625)

        self.assertEqual(d4.eq_dist(2), 0.25)
        self.assertEqual(d4.eq_dist(d4), 0.25)

        self.assertEqual(d4.ne_dist(2), 0.75)
        self.assertEqual(d4.ne_dist(d4), 0.75)

    def testCdfComparison(self):
        d4 = Cdf.from_seq(range(1, 5))
        self.assertEqual(d4.gt_dist(2), 0.5)
        self.assertEqual(d4.gt_dist(d4), 0.375)

        self.assertEqual(d4.lt_dist(2), 0.25)
        self.assertEqual(d4.lt_dist(d4), 0.375)

        self.assertEqual(d4.ge_dist(2), 0.75)
        self.assertEqual(d4.ge_dist(d4), 0.625)

        self.assertEqual(d4.le_dist(2), 0.5)
        self.assertEqual(d4.le_dist(d4), 0.625)

        self.assertEqual(d4.eq_dist(2), 0.25)
        self.assertEqual(d4.eq_dist(d4), 0.25)

        self.assertEqual(d4.ne_dist(2), 0.75)
        self.assertEqual(d4.ne_dist(d4), 0.75)

    def testSurvComparison(self):
        d4 = Surv.from_seq(range(1, 5))
        self.assertEqual(d4.gt_dist(2), 0.5)
        self.assertEqual(d4.gt_dist(d4), 0.375)

        self.assertEqual(d4.lt_dist(2), 0.25)
        self.assertEqual(d4.lt_dist(d4), 0.375)

        self.assertEqual(d4.ge_dist(2), 0.75)
        self.assertEqual(d4.ge_dist(d4), 0.625)

        self.assertEqual(d4.le_dist(2), 0.5)
        self.assertEqual(d4.le_dist(d4), 0.625)

        self.assertEqual(d4.eq_dist(2), 0.25)
        self.assertEqual(d4.eq_dist(d4), 0.25)

        self.assertEqual(d4.ne_dist(2), 0.75)
        self.assertEqual(d4.ne_dist(d4), 0.75)

    def testHazardComparison(self):
        d4 = Hazard.from_seq(range(1, 5))
        self.assertEqual(d4.gt_dist(2), 0.5)
        self.assertEqual(d4.gt_dist(d4), 0.375)

        self.assertEqual(d4.lt_dist(2), 0.25)
        self.assertEqual(d4.lt_dist(d4), 0.375)

        self.assertEqual(d4.ge_dist(2), 0.75)
        self.assertEqual(d4.ge_dist(d4), 0.625)

        self.assertEqual(d4.le_dist(2), 0.5)
        self.assertEqual(d4.le_dist(d4), 0.625)

        self.assertEqual(d4.eq_dist(2), 0.25)
        self.assertEqual(d4.eq_dist(d4), 0.25)

        self.assertEqual(d4.ne_dist(2), 0.75)
        self.assertEqual(d4.ne_dist(d4), 0.75)

    def testCdf(self):
        # if the quantities are not numeric, you can use [] but not ()
        cdf = Cdf.from_seq(list("allen"))
        self.assertAlmostEqual(cdf["a"], 0.2)
        self.assertAlmostEqual(cdf["e"], 0.4)
        self.assertAlmostEqual(cdf["l"], 0.8)
        self.assertAlmostEqual(cdf["n"], 1.0)

        t = [1, 2, 2, 3, 5]
        cdf = Cdf.from_seq(t)

        # () uses forward to interpolate
        self.assertEqual(cdf(0), 0)
        self.assertAlmostEqual(cdf(1), 0.2)
        self.assertAlmostEqual(cdf(2), 0.6)
        self.assertAlmostEqual(cdf(3), 0.8)
        self.assertAlmostEqual(cdf(4), 0.8)
        self.assertAlmostEqual(cdf(5), 1)
        self.assertAlmostEqual(cdf(6), 1)

        xs = range(-1, 7)
        ps = cdf(xs)
        for p1, p2 in zip(ps, [0, 0, 0.2, 0.6, 0.8, 0.8, 1, 1]):
            self.assertAlmostEqual(p1, p2)

        self.assertEqual(cdf.inverse(0), 1)
        self.assertEqual(cdf.inverse(0.1), 1)
        self.assertEqual(cdf.inverse(0.2), 1)
        self.assertEqual(cdf.inverse(0.3), 2)
        self.assertEqual(cdf.inverse(0.4), 2)
        self.assertEqual(cdf.inverse(0.5), 2)
        self.assertEqual(cdf.inverse(0.6), 2)
        self.assertEqual(cdf.inverse(0.7), 3)
        self.assertEqual(cdf.inverse(0.8), 3)
        self.assertEqual(cdf.inverse(0.9), 5)
        self.assertEqual(cdf.inverse(0.99999), 5)
        self.assertEqual(cdf.inverse(1), 5)

        ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        qs = cdf.inverse(ps)
        self.assertTrue((qs == [1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5]).all())

        np.random.seed(42)
        xs = cdf.choice(7, replace=True)
        self.assertListEqual(xs.tolist(), [2, 5, 3, 2, 1, 1, 1])

    def testSurv(self):
        # if the quantities are not numeric, you can use [] but not ()
        surv = Surv.from_seq(list("allen"))
        self.assertAlmostEqual(surv["a"], 0.8)
        self.assertAlmostEqual(surv["e"], 0.6)
        self.assertAlmostEqual(surv["l"], 0.2)
        self.assertAlmostEqual(surv["n"], 0)

        # test unnormalized
        t = [1, 2, 2, 3, 5]
        surv = Surv.from_seq(t, normalize=False)
        self.assertListEqual(list(surv), [4, 2, 1, 0])

        res = surv([0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(list(res), [5.0, 4.0, 2.0, 1.0, 1.0, 0.0, 0.0])

        res = surv.inverse([0, 1, 2, 3, 4, 5])
        self.assertListEqual(list(res), [5, 3, 2, 2, 1, -np.inf])

        # test normalized
        # () uses forward to interpolate
        surv = Surv.from_seq(t)
        self.assertEqual(surv(0), 1)
        self.assertAlmostEqual(surv(1), 0.8)
        self.assertAlmostEqual(surv(2), 0.4)
        self.assertAlmostEqual(surv(3), 0.2)
        self.assertAlmostEqual(surv(4), 0.2)
        self.assertAlmostEqual(surv(5), 0)
        self.assertAlmostEqual(surv(6), 0)

        xs = range(-1, 7)
        ps = surv(xs)
        for p1, p2 in zip(ps, [1, 1, 0.8, 0.4, 0.2, 0.2, 0, 0]):
            self.assertAlmostEqual(p1, p2)

        self.assertTrue(np.isnan(surv.inverse(-0.1)))
        self.assertEqual(surv.inverse(0), 5)
        self.assertEqual(surv.inverse(0.1), 5)
        self.assertEqual(surv.inverse(0.2), 3)
        self.assertEqual(surv.inverse(0.3), 3)
        self.assertEqual(surv.inverse(0.4), 2)
        self.assertEqual(surv.inverse(0.5), 2)
        self.assertEqual(surv.inverse(0.6), 2)
        self.assertEqual(surv.inverse(0.7), 2)
        self.assertEqual(surv.inverse(0.8), 1)
        self.assertEqual(surv.inverse(0.9), 1)
        self.assertEqual(surv.inverse(1), -np.inf)

        ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        qs = surv.inverse(ps)
        self.assertTrue((qs == [5, 5, 3, 3, 2, 2, 2, 2, 1, 1, -np.inf]).all())

        np.random.seed(42)
        xs = surv.choice(7, replace=True)
        self.assertListEqual(xs.tolist(), [2, 5, 3, 2, 1, 1, 1])

        # if the survival function starts at p[0]=1,
        # check that inverse() does the right thing
        surv[-2] = 1
        self.assertEqual(surv.inverse(1), -2)

    def testNormalize(self):
        t = [0, 1, 2, 3, 3, 4, 4, 4, 5]

        pmf = Pmf.from_seq(t, normalize=False)
        total = pmf.normalize()
        self.assertAlmostEqual(total, 9)
        self.assertAlmostEqual(pmf[3], 0.22222222)

        cdf = Cdf.from_seq(t, normalize=False)
        total = cdf.normalize()
        self.assertAlmostEqual(total, 9)
        self.assertAlmostEqual(cdf(3), 0.55555555)

    def testHazard(self):
        t = [1, 2, 2, 3, 5]
        haz = Hazard.from_seq(t)

        # () uses forward to interpolate
        self.assertAlmostEqual(haz(1), 0.2)
        self.assertAlmostEqual(haz(2), 0.5)
        self.assertAlmostEqual(haz(3), 0.5)
        self.assertAlmostEqual(haz(4), 0)
        self.assertAlmostEqual(haz(5), 1.0)
        self.assertAlmostEqual(haz(6), 0)

        xs = [0, 1, 2, 3, 4, 5, 6]
        res = haz(xs)
        for x, y in zip(res, [0, 0.2, 0.5, 0.5, 0, 1, 0]):
            self.assertAlmostEqual(x, y)

        cdf = Cdf.from_seq(t)
        haz2 = cdf.make_hazard()
        res = haz2(xs)
        for x, y in zip(res, [0, 0.2, 0.5, 0.5, 0, 1, 0]):
            self.assertAlmostEqual(x, y)

        surv = Surv.from_seq(t)
        haz3 = surv.make_hazard()
        res = haz3(xs)
        for x, y in zip(res, [0, 0.2, 0.5, 0.5, 0, 1, 0]):
            self.assertAlmostEqual(x, y)

    def testPmfFromCdf(self):
        t = [1, 2, 2, 3, 5]
        pmf = Pmf.from_seq(t)
        cdf = Cdf.from_seq(t)
        pmf2 = cdf.make_pmf()
        self.almost_equal_dist(pmf, pmf2)

    def testCredible(self):
        t = np.arange(101)
        pmf = Pmf.from_seq(t)
        cdf = pmf.make_cdf()
        ci = pmf.credible_interval(0.9)
        self.assertListEqual(list(ci), [5, 95])
        ci = cdf.credible_interval(0.9)
        self.assertListEqual(list(ci), [5, 95])

    def testMinMax(self):
        pmf = Pmf.from_seq([1, 2, 3])
        pmf2 = pmf.max_dist(2)
        ans = Pmf([1 / 9, 3 / 9, 5 / 9], pmf.index)
        self.almost_equal_dist(pmf2, ans)
        pmf3 = pmf.min_dist(2)
        ans = Pmf([5 / 9, 3 / 9, 1 / 9], pmf.index)
        self.almost_equal_dist(pmf3, ans)

    def testConversionFunctions(self):
        t = [1, 2, 2, 3, 5, 5, 7, 10]
        pmf = Pmf.from_seq(t)
        cdf = Cdf.from_seq(t)
        surv = Surv.from_seq(t)
        haz = Hazard.from_seq(t)

        cdf2 = pmf.make_cdf()
        self.almost_equal_dist(cdf, cdf2)

        surv2 = pmf.make_surv()
        self.almost_equal_dist(surv, surv2)

        haz2 = pmf.make_hazard()
        self.almost_equal_dist(haz, haz2)

        surv3 = haz2.make_surv()
        self.almost_equal_dist(surv, surv3)

        cdf3 = haz2.make_cdf()
        self.almost_equal_dist(cdf, cdf3)

        pmf3 = haz2.make_pmf()
        self.almost_equal_dist(pmf, pmf3)

    def testUnnormalized(self):
        t = [1, 2, 2, 4, 5]
        pmf = Pmf.from_seq(t, normalize=False)
        cdf = pmf.make_cdf()
        self.assertListEqual(list(cdf), [1, 3, 4, 5])

        surv = pmf.make_surv()
        self.assertListEqual(list(surv), [4, 2, 1, 0])

        cdf2 = surv.make_cdf()
        self.assertListEqual(list(cdf), list(cdf2))

        haz = pmf.make_hazard()
        self.assertListEqual(list(haz), [0.2, 0.5, 0.5, 1.0])

        pmf2 = haz.make_pmf()
        self.assertListEqual(list(pmf), list(pmf2))

    def testKaplanMeier(self):
        complete = [1, 3, 6]
        ongoing = [2, 3, 5, 7]

        pmf_complete = Pmf.from_seq(complete, normalize=False)
        pmf_ongoing = Pmf.from_seq(ongoing, normalize=False)

        res = pmf_complete + pmf_ongoing
        self.assertListEqual(list(res), [1, 1, 2, 1, 1, 1])

        res = pmf_complete - pmf_ongoing
        self.assertListEqual(list(res), [1.0, -1.0, 0.0, -1.0, 1.0, -1.0])

        res = pmf_complete * pmf_ongoing
        self.assertListEqual(list(res), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        res = pmf_complete / pmf_ongoing
        self.assertListEqual(list(res), [np.inf, 0.0, 1.0, 0.0, np.inf, 0.0])

        surv_complete = pmf_complete.make_surv()
        surv_ongoing = pmf_ongoing.make_surv()

        done = pmf_complete + pmf_ongoing

        s1 = surv_complete(done.index)
        self.assertListEqual(list(s1), [2.0, 2.0, 1.0, 1.0, 0.0, 0.0])

        s2 = surv_ongoing(done.index)
        self.assertListEqual(list(s2), [4.0, 3.0, 2.0, 1.0, 1.0, 0.0])

        at_risk = done + s1 + s2
        self.assertListEqual(list(at_risk), [7.0, 6.0, 5.0, 3.0, 2.0, 1.0])

        haz = pmf_complete / at_risk
        self.assertListEqual(
            list(haz), [0.14285714285714285, 0.0, 0.2, 0.0, 0.5, 0.0]
        )

    def almost_equal_dist(self, dist1, dist2):
        for x in dist1.qs:
            self.assertAlmostEqual(dist1[x], dist2[x])

    def testCopy(self):
        t = [1, 2, 2, 3, 5]
        pmf = Pmf.from_seq(t)

        pmf2 = pmf.copy()
        for x in pmf.qs:
            self.assertAlmostEqual(pmf[x], pmf2[x])

        cdf = pmf.make_cdf()
        cdf2 = cdf.copy()
        for x in cdf.qs:
            self.assertAlmostEqual(cdf[x], cdf2[x])

    def testPlotting(self):
        t = [1, 2, 2, 3, 5]

        pmf = Pmf.from_seq(t)
        pmf.bar()

        haz = Hazard.from_seq(t)
        haz.bar()


if __name__ == "__main__":
    unittest.main()
