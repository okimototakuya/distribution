import sys
sys.path.append('.')
import numpy as np
import gauss


class Conf():
    '''
    gauss.pyで扱う分布のパラメータを管理するクラス
    '''

    def __init__(self, input_z):
        '''
        コンストラクタ

        Notes
        -----
        分布について、各パラメータの設定はここで行う。
        '''
        ### 多変量ガウス分布
        #self.__mu = [-1, 1]                  # 平均値
        #self.__sigma = [[2, 1], [1, 3]]     # 分散/標準偏差
        ##mu = 0                     # 通った
        ##sigma = 1
        ##mu = [0, 0]                # 通った
        ##sigma = [[2, 1], [1, 3]]
        ##mu = [0]                    # 通らない
        ##sigma = [1]
        #self.__Z = gauss.multidim_gauss(input_z.T, mu=np.matrix(self.__mu).T, sigma=np.matrix(self.__sigma))

        ## 多変量混合ガウス分布
        self.__gauss_z_a = gauss.multidim_gauss(input_z.T, mu=np.matrix([-2,2]).T, sigma=np.matrix([[1,0],[0,1]]))      # 多変量単峰ガウス分布 (gauss.multidim_gauss)
        self.__gauss_z_b = gauss.multidim_gauss(input_z.T, mu=np.matrix([3,-3]).T, sigma=np.matrix([[1,0],[0,1]]))
        self.__Z = gauss.mixed_gauss(input_z, (self.__gauss_z_a, 3/4), (self.__gauss_z_b, 1/4))                         # 混合率

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
