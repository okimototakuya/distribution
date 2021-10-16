import sys
import unittest
import sampling
sys.path.append('../distribution')
import gauss


class TestSampling(unittest.TestCase):
    '''
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_sample_mixed_gauss(self):
        '''
        関数sampling.sample_mixed_gaussと関数sampling.metropolisによるサンプリング結果が概ね一致するかテスト.

        Notes
        -----
        - 値はベタ書き
        - [テスト方法1]: self.assertAlmostEqualでテスト
        - [テスト方法2]: プロダクトコード (sampling.py) のmain関数にて、ヒストグラムをプロットして分布形を確認
        '''
        p = lambda theta: gauss.mixed_gauss(theta,  \
                                            (gauss.gauss(theta, mu=0, sigma=1), 3/4),   \
                                            (gauss.gauss(theta, mu=3, sigma=1), 1/4),   # 単変量混合ガウス分布
                                           )
        self.assertAlmostEqual(sampling.metropolis(p), sampling.sample_mixed_gauss(p))

    def test_sample_mixed_gauss(self):
        '''
        関数sampling.sample_mixed_gaussと関数sampling.metropolisによるサンプリング結果が概ね一致するかテスト.

        Notes
        -----
        - パラメータはリスト型で取得
        - 単峰ガウス分布の混合は、2つまで対応
        - 混合率の和が1でないとき、例外を発生 (実際に例外を発生させてテスト済
        '''
        mu = [0, 3]
        sigma = [1, 1]
        rate = [3/4, 1/4]
        #rate = [1/4, 1/4]
        p = lambda theta: gauss.mixed_gauss(theta,  \
                                            (gauss.gauss(theta, mu=mu[0], sigma=sigma[0]), rate[0]),   \
                                            (gauss.gauss(theta, mu=mu[1], sigma=sigma[1]), rate[1]),   # 単変量混合ガウス分布
                                           )
        self.assertAlmostEqual(sampling.metropolis(p), sampling.sample_mixed_gauss(mu, sigma, rate))


if __name__ == '__main__':
    unittest.main()