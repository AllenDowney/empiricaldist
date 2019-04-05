"""This file contains code for use with "Think Bayes, 2nd edition",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import unittest
import random

from collections import Counter
import numpy as np

from distribution import Pmf, Cdf

class Test(unittest.TestCase):

    def testPmf(self):
        t = list('allen')
        pmf = Pmf.from_seq(t)

        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf['l'], 0.4)

        pmf = Pmf(pmf)
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf['l'], 0.4)

        pmf = Pmf(Counter(t))
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf['l'], 2)

        pmf2 = pmf.copy()
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf['l'], 2)

        np.random.seed(42)
        pmf.normalize()
        xs = pmf.choice(7, replace=True)
        self.assertListEqual(xs.tolist(), ['l', 'n', 'e', 'l', 'a', 'a', 'a'])

    def testStats(self):
        pmf = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(pmf.mean(), 3.5)
        self.assertAlmostEqual(pmf.var(), 2.91666666)
        self.assertAlmostEqual(pmf.std(), 1.70782512)
        self.assertAlmostEqual(pmf.median(), 3)

        cdf = pmf.make_cdf()
        self.assertAlmostEqual(cdf.mean(), 3.5)
        self.assertAlmostEqual(cdf.var(), 2.91666666)
        self.assertAlmostEqual(cdf.std(), 1.70782512)
        self.assertAlmostEqual(cdf.median(), 3)

    def testAdd(self):
        pmf = Pmf.from_seq([1, 2, 3, 4, 5, 6])

        pmf1 = pmf + 1
        self.assertAlmostEqual(pmf1.mean(), 4.5)

        pmf1 = 1 + pmf
        self.assertAlmostEqual(pmf1.mean(), 4.5)

        pmf2 = pmf + pmf
        self.assertAlmostEqual(pmf2.mean(), 7.0)

    def testSub(self):
        pmf = Pmf.from_seq([1, 2, 3, 4, 5, 6])

        pmf3 = pmf - 1
        self.assertAlmostEqual(pmf3.mean(), 2.5)

        # TODO: implement rsub correctly
        #pmf3 = 1 - pmf
        #self.assertAlmostEqual(pmf3.mean(), -2.5)

        pmf4 = pmf - pmf
        self.assertAlmostEqual(pmf4.mean(), 0)

    def test_joint(self):
        pmf1 = Pmf.from_seq([1, 2, 2])
        pmf2 = Pmf.from_seq([1, 2, 3])

        joint = Pmf.make_joint(pmf1, pmf2)

        mar1 = joint.marginal(0)
        mar2 = joint.marginal(1)
        self.assertAlmostEqual(mar1.mean(), pmf1.mean())
        self.assertAlmostEqual(mar2.mean(), pmf2.mean())

        cond1 = joint.conditional(0, 1, 1)
        cond2 = joint.conditional(1, 0, 1)
        self.assertAlmostEqual(cond1.mean(), pmf1.mean())
        self.assertAlmostEqual(cond2.mean(), pmf2.mean())


    def testCamparison(self):
        pmf1 = Pmf.from_seq([1, 2, 3, 4, 5, 6])
        pmf2 = Pmf.from_seq([1, 2, 3, 4])

        self.assertAlmostEqual(pmf1 == 3, 1 / 6)
        self.assertAlmostEqual(pmf1 != 3, 5 / 6)
        self.assertAlmostEqual(pmf1 > 3, 3 / 6)
        self.assertAlmostEqual(pmf1 >= 3, 4 / 6)
        self.assertAlmostEqual(pmf1 < 3, 2 / 6)
        self.assertAlmostEqual(pmf1 <= 3, 3 / 6)

        self.assertAlmostEqual(pmf1 == pmf2, 1/6)
        self.assertAlmostEqual(pmf1 != pmf2, 5/6)
        self.assertAlmostEqual(pmf1 > pmf2, 0.5833333)
        self.assertAlmostEqual(pmf1 >= pmf2, 3/4)
        self.assertAlmostEqual(pmf1 < pmf2, 1/4)
        self.assertAlmostEqual(pmf1 <= pmf2, 0.41666666)

    def testPmfComparison(self):
        d4 = Pmf.from_seq(range(1,5))
        self.assertEqual(d4.gt(2), 0.5)
        self.assertEqual(d4.gt(d4), 0.375)

        self.assertEqual(d4.lt(2), 0.25)
        self.assertEqual(d4.lt(d4), 0.375)

        self.assertEqual(d4.ge(2), 0.75)
        self.assertEqual(d4.ge(d4), 0.625)

        self.assertEqual(d4.le(2), 0.5)
        self.assertEqual(d4.le(d4), 0.625)

        self.assertEqual(d4.eq(2), 0.25)
        self.assertEqual(d4.eq(d4), 0.25)

        self.assertEqual(d4.ne(2), 0.75)
        self.assertEqual(d4.ne(d4), 0.75)

    def testCdf(self):
        t = [1, 2, 2, 3, 5]
        cdf = Cdf.from_seq(t)

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
        self.assertEqual(cdf.inverse(1), 5)

        ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        qs = cdf.inverse(ps)
        self.assertTrue((qs == [1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5]).all())

        np.random.seed(42)
        xs = cdf.choice(7, replace=True)
        self.assertListEqual(xs.tolist(), [2, 5, 3, 2, 1, 1, 1])
        
    def testPmfFromCdf(self):
        t = [1, 2, 2, 3, 5]
        pmf = Pmf.from_seq(t)
        cdf = Cdf.from_seq(t)
        pmf2 = cdf.make_pmf()
        for x in pmf.qs:
            self.assertAlmostEqual(pmf[x], pmf2[x])

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


if __name__ == "__main__":
    unittest.main()
