import sys
import unittest
import numpy as np
sys.path.append('../main')
import gauss


class TestMixedGauss(unittest.TestCase):
    '''
    gaussモジュールについて、関数mixed_gaussをテスト
    '''
    @property
    def sample_size(self, input_sample_size):
        self.__sample_size = input_sample_size

    @sample_size.getter
    def sample_size(self):
        return self.__sample_size

    def setUp(self):
        self.__sample_size = 100
        self.x = np.linspace(-3, 3, self.sample_size)       # x軸がシーケンス (例.プロット)
        #self.x = 0                                         # x軸がスカラー (例.サンプリング)

    def tearDown(self):
        pass

    def test_mixed_gauss_given_2_standard_gauss_who_has_mixture_ratio(self):
        '''
        関数mixed_gaussについて、関数gaussの出力と関数mixed_gaussの出力 (np.ndarray型) が一致するか確認.
        ただし、関数mixed_gaussが受け取る引数は、xと2つの混合率を持つ標準正規分布のみ。
        '''
        gauss_y = gauss.gauss(self.x, 0, 1)
        mixed_gauss_y = gauss.mixed_gauss(self.x, (gauss_y, 1/2), (gauss_y, 1/2))
        np.testing.assert_array_equal(gauss_y, mixed_gauss_y)   # 通る.

    def test_mixed_gauss_given_2_standard_multidim_gauss_whose_parameter_is_solo(self):
        '''
        関数mixed_gaussについて、関数multidim_gaussの出力と関数mixed_gaussの出力 (np.ndarray型) が一致するか確認.
        ただし、関数mixed_gaussが受け取る引数は、xと2つの混合率を持つ単変量標準正規分布のみ。
        '''
        gauss_y = gauss.multidim_gauss(self.x, np.matrix([0]).T, np.matrix([1]))
        mixed_gauss_y = gauss.mixed_gauss(self.x, (gauss_y, 1/2), (gauss_y, 1/2))
        np.testing.assert_array_equal(gauss_y, mixed_gauss_y)   # 通る.

    def test_mixed_gauss_given_2_standard_multidim_gauss_whose_parameter_is_multi(self):
        '''
        関数mixed_gaussについて、関数multidim_gaussの出力と関数mixed_gaussの出力 (np.ndarray型) が一致するか確認.
        ただし、関数mixed_gaussが受け取る引数は、xと2つの混合率を持つ多変量標準正規分布のみ。
        '''
        gauss_y = gauss.multidim_gauss(self.x, np.matrix([0,0]).T, np.matrix([[1,0],[0,1]]))
        mixed_gauss_y = gauss.mixed_gauss(self.x, (gauss_y, 1/2), (gauss_y, 1/2))
        np.testing.assert_array_equal(gauss_y, mixed_gauss_y)   # 通る.


if __name__ == '__main__':
    unittest.main()
