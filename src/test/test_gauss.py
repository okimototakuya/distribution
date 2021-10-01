import sys
import unittest
import numpy as np
sys.path.append('../main')
import gauss


class TestGauss(unittest.TestCase):

    @property
    def sample_size(self, input_sample_size):
        self.__sample_size = input_sample_size

    @sample_size.getter
    def sample_size(self):
        return self.__sample_size

    def setUp(self):
        self.__sample_size = 100
        self.x = np.linspace(-3, 3, self.sample_size)

    def tearDown(self):
        pass

    def test_gauss_gauss_return_type(self):
        '''
        gaussモジュールについて、関数gaussの返り値の型を確認.
        '''
        gauss_y = gauss.gauss(self.x, 0, 1)
        self.assertIsInstance(gauss_y, np.ndarray)  # numpy.ndarray配列型かアサーション


if __name__ == '__main__':
    unittest.main()
