import sys
sys.path.append('.')
import numpy as np
import gauss


class Conf():
    '''
    '''

    def __init__(self, input_z):
        '''
        コンストラクタ

        Notes
        -----
        各パラメータの設定は、ここで行う。
        '''
        ## 多変量ガウス分布
        self.__mu = [-1, 1]                  # 平均値
        self.__sigma = [[2, 1], [1, 3]]     # 分散/標準偏差
        #mu = 0                     # 通った
        #sigma = 1
        #mu = [0, 0]                # 通った
        #sigma = [[2, 1], [1, 3]]
        #mu = [0]                    # 通らない
        #sigma = [1]
        self.__Z = gauss.multidim_gauss(input_z.T, mu=np.matrix(self.__mu).T, sigma=np.matrix(self.__sigma))        # ガウス分布 (gauss.multidim_gauss)

    ## ガウス分布
    @property
    def mu(self):
        '''
        平均値
        '''
        return self.__mu

    @property
    def sigma(self):
        '''
        分散/標準偏差
        '''
        return self.__sigma

    @property
    def Z(self):
        '''
        密度関数の出力
        '''
        return self.__Z


def main():
    pass

if __name__ == '__main__':
    main()
