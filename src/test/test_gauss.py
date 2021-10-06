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

    def _test_mixed_gauss_given_1_standard_gauss(self):
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
        '''
        ## 配列同士の差
        #y = gauss.gauss(self.x, 0, np.sqrt(2))
        #z = gauss.multidim_gauss(self.x.T, mu=np.matrix([0]).T, sigma=np.matrix([2]))
        #print(y-z)
        ## 関数gauss.gaussについて
        #print('gauss.gaussが返す値:{ret_val}'.format(ret_val=y))
        #print('gauss.gaussが返す型:{ret_type}'.format(ret_type=type(y)))
        #print('gauss.gaussが返す値の大きさ:{ret_val_size}'.format(ret_val_size=len(y)))
        ## 関数gauss.multidim_gaussについて
        #print('gauss.multidim_gaussが返す値:{ret_val}'.format(ret_val=z))
        #print('gauss.multidim_gaussが返す型:{ret_type}'.format(ret_type=type(z)))
        #print('gauss.multidim_gaussが返す値の大きさ:{ret_val_size}'.format(ret_val_size=len(z)))
        ## アサーション
        #np.testing.assert_array_equal(gauss.gauss(self.x, 0, 1), gauss.multidim_gauss(self.x.T, mu=np.matrix([0]).T, sigma=np.matrix([1])))                   #1. 単変量標準正規分布同士 (通る)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 0, 1), gauss.multidim_gauss(self.x.T, mu=np.matrix([0, 0]).T, sigma=np.matrix([[1, 0], [0, 1]])))   #2. 単変量 " と2変量 " (通らない)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 34, 1), gauss.multidim_gauss(self.x.T, mu=np.matrix([34]).T, sigma=np.matrix([1])))                 #3. 期待値mu not eq 0 (通る)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 0, np.sqrt(2)), gauss.multidim_gauss(self.x.T, mu=np.matrix([0]).T, sigma=np.matrix([2])))          #4. 分散sigma not eq 1 (通らない)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 45, 2), gauss.multidim_gauss(self.x.T, mu=np.matrix([45]).T, sigma=np.matrix([2])))                 #5. 3かつ4 (通らない)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 3, np.sqrt(2)), gauss.multidim_gauss(self.x.T, mu=np.matrix([3]).T, sigma=np.matrix([2])))          #6. gauss.gaussの方は標準偏差を与える (通らないが、値はほぼ一致. おそらく数値誤差の分)
        np.testing.assert_array_almost_equal(gauss.gauss(self.x, 0, np.sqrt(2)), gauss.multidim_gauss(self.x.T, mu=np.matrix([0]).T, sigma=np.matrix([2])))    #7. assertAlmostEqualでテスト(通る)
        ## assertAlmostEqualについて、学習用テスト
        ### スカラー
        #self.assertAlmostEqual(1, 0.99999996)   # assertAlmostEqualの仕様: 有効数字が小数点以下8桁 かつ 8桁目が6以上で通る.
        ### シーケンス
        #self.assertAlmostEqual([1, 2, 3], [1, 2, 3])                                                                               # 通る
        #self.assertAlmostEqual([1, 2, 3], [1, 2, 9])                                                                               # 通らない (標準リストだと、-演算がサポートされていないため.)
        #np.testing.assert_array_almost_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))                                             # 通る
        #np.testing.assert_array_almost_equal(np.array([i for i in range(100)]), np.array([i for i in range(100)][:99]+[99]))       # 通る (要素の非合致率1% かつ その値の差が1. しきい値は不明.)


if __name__ == '__main__':
    unittest.main()
