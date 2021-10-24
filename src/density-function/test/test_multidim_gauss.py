import sys
import unittest
import numpy as np
sys.path.append('../main')
import gauss


class TestMultidimGauss(unittest.TestCase):
    '''
    gaussモジュールについて、関数multidim_gaussをテスト
    '''
    @property
    def sample_size(self, input_sample_size):
        self.__sample_size = input_sample_size

    @sample_size.getter
    def sample_size(self):
        return self.__sample_size

    def setUp(self):
        self.__sample_size = 100
        #self.x = np.linspace(-3, 3, self.sample_size)       # x軸がシーケンス (例.プロット)
        self.x = 0                                         # x軸がスカラー (例.サンプリング)

    def tearDown(self):
        pass

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
        np.testing.assert_array_equal(gauss.gauss(self.x, 0, 1), gauss.multidim_gauss(self.x, mu=np.matrix([0]).T, sigma=np.matrix([1])))                   #1. 単変量標準正規分布同士
        #np.testing.assert_array_equal(gauss.gauss(self.x, 0, 1), gauss.multidim_gauss(self.x, mu=np.matrix([0, 0]).T, sigma=np.matrix([[1, 0], [0, 1]])))   #2. 単変量 " と2変量 " (通らない)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 34, 1), gauss.multidim_gauss(self.x, mu=np.matrix([34]).T, sigma=np.matrix([1])))                 #3. 期待値mu not eq 0 (通る)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 0, np.sqrt(2)), gauss.multidim_gauss(self.x, mu=np.matrix([0]).T, sigma=np.matrix([2])))          #4. 分散sigma not eq 1 (通らない)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 45, 2), gauss.multidim_gauss(self.x, mu=np.matrix([45]).T, sigma=np.matrix([2])))                 #5. 3かつ4 (通らない)
        #np.testing.assert_array_equal(gauss.gauss(self.x, 3, np.sqrt(2)), gauss.multidim_gauss(self.x, mu=np.matrix([3]).T, sigma=np.matrix([2])))          #6. gauss.gaussの方は標準偏差を与える (通らないが、値はほぼ一致. おそらく数値誤差の分)
        #np.testing.assert_array_almost_equal(gauss.gauss(self.x, 0, np.sqrt(2)), gauss.multidim_gauss(self.x, mu=np.matrix([0]).T, sigma=np.matrix([2])))    #7. assertAlmostEqualでテスト(通る)
        ## assertAlmostEqualについて、学習用テスト
        ### スカラー
        #self.assertAlmostEqual(1, 0.99999996)   # assertAlmostEqualの仕様: 有効数字が小数点以下8桁 かつ 8桁目が6以上で通る.
        ### シーケンス
        #self.assertAlmostEqual([1, 2, 3], [1, 2, 3])                                                                               # 通る
        #self.assertAlmostEqual([1, 2, 3], [1, 2, 9])                                                                               # 通らない (標準リストだと、-演算がサポートされていないため.)
        #np.testing.assert_array_almost_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))                                             # 通る
        #np.testing.assert_array_almost_equal(np.array([i for i in range(100)]), np.array([i for i in range(100)][:99]+[99]))       # 通る (要素の非合致率1% かつ その値の差が1. しきい値は不明.)

    def test_multidim_gauss_given_x_whose_type_is_int(self):
        '''
        gaussモジュールについて、関数multidim_gaussに1次元のx軸(int)を与えた場合に、関数gaussと関数multidim_gaussの返す値が一致するかテスト.

        Notes
        -----
        - 応用例: sampling.py (サンプリングアルゴリズム) で密度関数扱いする時など
        '''
        np.testing.assert_array_equal(gauss.gauss(self.x, 0, 1), gauss.multidim_gauss(self.x, mu=np.matrix([0]).T, sigma=np.matrix([1])))

    def test_multidim_gauss_return_type_given_1_dimensional_parameter(self):
        '''
        gaussモジュールについて、関数multidim_gaussの返り値の型をテスト.
        ただし、multidim_gaussに与えるパラメータは1次元.
        '''
        ret_val = gauss.multidim_gauss(self.x, mu=np.matrix([0]).T, sigma=np.matrix([1]))
        self.assertIsInstance(ret_val[0], float)

    def test_multidim_gauss_return_type_given_2_dimensional_parameter(self):
        '''
        gaussモジュールについて、関数multidim_gaussの返り値の型をテスト.
        ただし、multidim_gaussに与えるパラメータは2次元.
        '''
        ret_val = gauss.multidim_gauss(self.x, mu=np.matrix([0, 0]).T, sigma=np.matrix([[1, 0], [0, 1]]))
        self.assertIsInstance(ret_val[0], float)


if __name__ == '__main__':
    unittest.main()
