class Conf():
    '''
    '''

    def __init__(self):
        '''
        コンストラクタ

        Notes
        -----
        各パラメータの設定は、ここで行う。
        '''
        self.__sample_size = 100            # 標本列のサンプルサイズ
        # FIXME: 2021.10.14: どの分布を扱うか
        self.__mu = [0, 0]                  # 平均値
        self.__sigma = [[2, 1], [1, 3]]     # 分散/標準偏差
        #mu = 0                     # 通った
        #sigma = 1
        #mu = [0, 0]                # 通った
        #sigma = [[2, 1], [1, 3]]
        #mu = [0]                    # 通らない
        #sigma = [1]

    @property
    def sample_size(self):
        '''
        標本列のサンプルサイズ
        '''
        return self.__sample_size

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


def main():
    pass

if __name__ == '__main__':
    main()
