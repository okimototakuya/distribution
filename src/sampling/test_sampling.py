import sys
import unittest
import numpy as np
import sampling
sys.path.append('../distribution')
import gauss

class TestSampling(unittest.TestCase):
    '''
    sampling.pyについてテスト
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_sample_mixed_gauss_set_parameter_in_advance(self):
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

    def _test_sample_mixed_gauss_given_list_who_has_2_elements(self):
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

    def _test_sample_mixed_gauss_given_list_who_has_3_elements(self):
        '''
        関数sampling.sample_mixed_gaussと関数sampling.metropolisによるサンプリング結果が概ね一致するかテスト.

        Notes
        -----
        - パラメータはリスト型で取得
        - 単峰ガウス分布の混合は、3つまで対応
        - 混合率の和が1でないとき、例外を発生 (実際に例外を発生させてテスト済
        '''
        mu = [0, 3, 6]
        sigma = [1, 1, 1]
        rate = [2/4, 1/4, 1/4]
        p = lambda theta: gauss.mixed_gauss(theta,  \
                                            (gauss.gauss(theta, mu=mu[0], sigma=sigma[0]), rate[0]),   \
                                            (gauss.gauss(theta, mu=mu[1], sigma=sigma[1]), rate[1]),   \
                                            (gauss.gauss(theta, mu=mu[2], sigma=sigma[2]), rate[2]),   # 単変量混合ガウス分布
                                           )
        self.assertAlmostEqual(sampling.metropolis(p), sampling.sample_mixed_gauss(mu, sigma, rate))

    def _test_sample_mixed_gauss_given_list_who_has_multi_elements(self):
        '''
        関数sampling.sample_mixed_gaussと関数sampling.metropolisによるサンプリング結果が概ね一致するかテスト.

        Notes
        -----
        - パラメータはリスト型で取得
        - 単峰ガウス分布の混合は、任意数対応
        - 混合率の和が1でないとき、例外を発生 (実際に例外を発生させてテスト済
        '''
        mu = [0, 3, 6, 9]
        sigma = [1, 1, 1, 1]
        rate = [1/5, 2/5, 1/5, 1/5]
        p = lambda theta: gauss.mixed_gauss(theta,  \
                                            (gauss.gauss(theta, mu=mu[0], sigma=sigma[0]), rate[0]),   \
                                            (gauss.gauss(theta, mu=mu[1], sigma=sigma[1]), rate[1]),   \
                                            (gauss.gauss(theta, mu=mu[2], sigma=sigma[2]), rate[2]),   \
                                            (gauss.gauss(theta, mu=mu[3], sigma=sigma[3]), rate[3]),   # 単変量混合ガウス分布
                                           )
        self.assertAlmostEqual(sampling.metropolis(p), sampling.sample_mixed_gauss(mu, sigma, rate))

    def test_sample_hmm_given_transiton_matrix_who_has_same_list_element_matches_gmm(self):
        '''
        関数sampling/sample_hmmについて、遷移行列 (GMMでの混合率) を同じリストを要素に持つ行列にした場合、
        サンプリング結果がGMMによるものと一致するかテスト

        Notes
        -----
        '''
        mu = [0, 10]
        sigma = [1, 1]
        list_rate = [9/10, 1/10]
        rate_hmm = [list_rate, list_rate]
        rate_gmm = list_rate
        sampling.sample_list = [10]                 # HMMのサンプリング
        sampling.sample_hmm(mu, sigma, rate_hmm)
        print('HMMのサンプリング')
        print(sampling.sample_list[-20:])
        sampling.sample_list = [10]                 # GMMのサンプリング
        sampling.sample_mixed_gauss(mu, sigma, rate_gmm)
        print('GMMのサンプリング')
        print(sampling.sample_list[-20:])
        #self.assertAlmostEqual(sampling.sample_hmm(mu, sigma, rate_hmm), sampling.sample_mixed_gauss(mu, sigma, rate_gmm))

    def _test_metropolis_given_multidim_density_function_whose_parameter_is_solo(self):
        '''
        sampling.metropolis (メトロポリス法) について、多次元の分布が適用できるかテスト.
        ただし、多変量の密度関数について、単変量のパラメータを与えた.
        '''
        # 多変量ガウス(に、単変量のパラメータを与える)
        p_multidim = lambda theta: gauss.multidim_gauss(theta,   \
                                               mu = np.matrix([0]).T,    \
                                               sigma = np.matrix([[1]]),  \
                                              )
        test_list_by_np = [np.random.normal() for _ in range(sampling.sample_size)]
        sampling.metropolis(p_multidim)     # 2021.10.24: FIXME: 一度はサンプリングできたが、できなくなった.
        print('test_list_by_np')
        print(test_list_by_np[:10])
        print('sampling.sample_list')
        print(sampling.sample_list[:10])
        # 2021.10.24: FIXME: アサーションの仕方を考える必要がある.
        self.assertAlmostEqual(test_list_by_np, sampling.sample_list)

    def _test_metropolis_given_multidim_density_function_whose_parameter_is_2dim(self):
        '''
        sampling.metropolis (メトロポリス法) について、多次元の分布が適用できるかテスト.
        ただし、多変量の密度関数について、2変量のパラメータを与えた.
        '''
        # 多変量ガウス
        p_multidim = lambda theta: gauss.multidim_gauss(theta,   \
                                               mu = np.matrix([0, 0]).T,    \
                                               sigma = np.matrix([[1, 0], [0, 1]]),  \
                                              )
        test_list_by_np = (np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], sampling.sample_size)).tolist()
        sampling.multidim_metropolis(p_multidim)
        print('test_list_by_np')
        print(test_list_by_np[:10])
        print('sampling.sample_list')
        print(sampling.sample_list[:10])
        # アサーションは通らないが、とりあえずサンプリングはできている.
        # 2021.10.24: FIXME: アサーションの仕方を考える必要がある.
        self.assertAlmostEqual(test_list_by_np, sampling.sample_list)


if __name__ == '__main__':
    unittest.main()
