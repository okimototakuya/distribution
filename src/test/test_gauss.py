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

    def _test_gauss_mixedgauss_parameter(self):
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
        np.testing.assert_array_equal(gauss_y, mixed_gauss_y)   # 通る.

    def test_multidim_gauss_given_1_dimensional_parameter(self):
        '''
        gaussモジュールについて、関数multidim_gaussに1次元の引数を与えた場合に、関数gaussと関数multidim_gaussの返す値が一致するかテスト.

        Notes
        -----
        予想外に見事に一致した！
        '''
        y = gauss.gauss(self.x, 0, 1)
        z = gauss.multidim_gauss(self.x.T, mu=np.matrix([0]).T, sigma=np.matrix([1]))
        ## 関数gauss.gauss
        #print('gauss.gaussが返す値:{ret_val}'.format(ret_val=y))
        #print('gauss.gaussが返す型:{ret_type}'.format(ret_type=type(y)))
        #print('gauss.gaussが返す値の大きさ:{ret_val_size}'.format(ret_val_size=len(y)))
        ## 関数gauss.multidim_gauss
        #print('gauss.multidim_gaussが返す値:{ret_val}'.format(ret_val=z))
        #print('gauss.multidim_gaussが返す型:{ret_type}'.format(ret_type=type(z)))
        #print('gauss.multidim_gaussが返す値の大きさ:{ret_val_size}'.format(ret_val_size=len(z)))
        np.testing.assert_array_equal(gauss.gauss(self.x, 0, 1), gauss.multidim_gauss(self.x.T, mu=np.matrix([0]).T, sigma=np.matrix([1])))

    def test_multidim_gauss_ret_val(self):
        '''
        関数multidim_gaussについて、返り値の詳細をテスト.

        Notes
        -----
        帰り値の値や型、大きさなど.
        '''
        z = gauss.multidim_gauss(self.x.T, mu=np.matrix([0, 0]).T, sigma=np.matrix([[1, 0], [0, 1]]))
        print('gauss.multidim_gaussが返す値:{ret_val}'.format(ret_val=z))
        print('gauss.multidim_gaussが返す型:{ret_type}'.format(ret_type=type(z)))
        print('gauss.multidim_gaussが返す値の大きさ:{ret_val_size}'.format(ret_val_size=len(z)))
        self.assertIsInstance(z, np.ndarray)    # 通る

if __name__ == '__main__':
    unittest.main()
