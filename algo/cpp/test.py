import unittest
import numpy as np
from algo.cpp.cseries import shift_forward, compute_ema


class MyTestCase(unittest.TestCase):
    def test_shift(self):
        ts = np.array([100, 200, 300, 400])
        xs = np.array([1., 2., 3., 4.])
        ys = shift_forward(ts, xs, 150)

        np.testing.assert_almost_equal(ys, np.array([3., 4., 4., 4.]))

    def test_ema(self):
        ts = np.array([1, 2, 3, 4])
        xs = np.array([1., 2., 3., 4.])
        ds = 1.0
        np.testing.assert_almost_equal(compute_ema(ts, xs, ds), np.array([1., 1.63212056, 2.49678528, 3.44699821]))


if __name__ == '__main__':
    unittest.main()
