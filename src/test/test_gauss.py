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

    def test_gauss_mixedgauss_parameter(self):
        '''
        gaussモジュールについて、関数mixed_gaussの引数をx (np.ndarray) とinput_gauss (np.ndarray) の2つに変更し、
        正しく動作 (np.ndarray型を返すかどうか) を確認.
        '''
        mixed_gauss_y = gauss.mixed_gauss(self.x, gauss.gauss(self.x, 0, 1))
        self.assertIsInstance(mixed_gauss_y, np.ndarray)  # numpy.ndarray配列型かアサーション

    def test_gauss_assertequal_gauss_mixed_gauss(self):
        '''
        関数mixed_gaussについて、関数gaussと関数mixed_gaussの出力 (np.ndarray型) が一致するか確認.
        ただし、関数mixed_gaussが受け取る引数は、xと1つの標準正規分布のみ。
        '''
        gauss_y = gauss.gauss(self.x, 0, 1)
        mixed_gauss_y = gauss.mixed_gauss(self.x,gauss_y)
        np.testing.assert_array_equal(gauss_y, mixed_gauss_y)


if __name__ == '__main__':
    unittest.main()
