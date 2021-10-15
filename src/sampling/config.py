import sys
sys.path.append('../distribution')
import gauss
import sampling

class Conf():
    '''
    sampling.pyで用いるパラメータを管理する.

    Notes
    -----
    FIXME: 2021.10.15
    - config.pyで、1.所望の分布, 2.サンプリング・アルゴリズムを管理しようと考えたが、Template Methodで考えた方が自然な気がする。
    - とりあえずプログラムは動いたが、sampling.pyのグローバル変数sampling_listが初期値のまま更新されたかった。
    '''
    def __init__(self):
        '''
        コンストラクタ

        Parameters
        -----
        - target_distribution: function型
            所望の分布の密度関数をfunction型で受け取る.
        - sampling_algorithm: 
            サンプリング・アルゴリズムを受け取る.
        '''
        ## 所望の分布
        self.__target_distribution = lambda theta: gauss.gauss(theta, mu=0, sigma=1)    # 単変量ガウス分布 ([＊]theta : intまたはnp.ndarray
        #self.__target_distribution = lambda theta: gauss.mixed_gauss(theta,  \
        #                                                             (gauss.gauss(theta, mu=0, sigma=1), 1/4),   \
        #                                                             (gauss.gauss(theta, mu=5, sigma=1), 1/4),   \
        #                                                             (gauss.gauss(theta, mu=3, sigma=1), 2/4))   # 単変量混合ガウス分布
        ## サンプリング・アルゴリズム
        #self.__sampling_algorithm = sampling.metropolis(self.__target_distribution)   # メトロポリス法
        self.__sampling_algorithm = sampling.metropolis_hastings(self.__target_distribution)   # メトロポリス・ヘイスティングス法

    @property
    def target_distribution(self):
        '''
        所望の分布を返す.
        '''
        return self.__target_distribution

    @property
    def sampling_algorithm(self):
        '''
        サンプリング・アルゴリズムを返す.
        '''
        return self.__sampling_algorithm

def main():
    pass

if __name__ == '__main__':
    main()
