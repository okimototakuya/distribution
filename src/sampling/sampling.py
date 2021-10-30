import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as cm
import config
sys.path.append('../density-function/main')
import gauss

sample_list = []
#sample_list.append(10)  # 初期値を適当に10と定めた
#sample_list.append([10, 10])  # 初期値を適当に10と定めた
sample_size = 10000    # サンプルサイズ

def metropolis(p):
    '''
    メトロポリス法を実装.
    メトロポリス・ヘイスティングス法の簡易版.
    代理分布: Normal(sample_list[i], 1)

    Parameters
    ------
    p : function
        所望の分布
    '''
    for i in range(sample_size):
        theta = np.random.normal(sample_list[i], 1) # 提案値
        a = min(1, p(theta) / p(sample_list[i]))

        u = np.random.rand() # 0 ~ 1 の一様乱数を生成

        if u < a: # 受諾
            sample_list.append(theta)
        else: # 拒否
            sample_list.append(sample_list[i])  # 1つ前の標本を保持

def multidim_metropolis(p):
    '''
    多変量メトロポリス法を実装.
    多変量メトロポリス・ヘイスティングス法の簡易版.
    代理分布: Normal(sample_list[i], [[1, 0], [0, 1]])

    Parameters
    ------
    p : function
        所望の分布
    '''
    for i in range(sample_size):
        theta = np.random.multivariate_normal(sample_list[i], [[1, 0], [0, 1]], 1).tolist()[0] # 提案値
        a = min(1, p(np.matrix(theta)) / p(np.matrix(sample_list[i])))

        u = np.random.rand() # 0 ~ 1 の一様乱数を生成

        if u < a: # 受諾
            sample_list.append(theta)
        else: # 拒否
            sample_list.append(sample_list[i])  # 1つ前の標本を保持

def metropolis_hastings(p):
    '''
    メトロポリス・ヘイスティングス法を実装.
    メトロポリス法と異なり、任意の代理分布を設定できる. (＊ただし、1つ前の標本に依存する.

    Parameters
    ------
    p : function
        所望の分布
    '''
    # 代理分布: 標準偏差sigmaのガウス分布の場合
    sigma = 1
    q = lambda x : np.random.normal(x, sigma)   # 代理分布に従う乱数
    p_arp = lambda x, mu : gauss.gauss(x, mu, sigma)    # 代理分布の密度関数

    for i in range(sample_size):
        theta = q(sample_list[i])   # 提案値
        a = (p(theta)*p_arp(sample_list[i], theta)) / (p(sample_list[i])*p_arp(theta, sample_list[i]))

        u = np.random.rand() # 0 ~ 1 の一様乱数を生成

        if u < a: # 受諾
            sample_list.append(theta)
        else: # 拒否
            sample_list.append(sample_list[i])  # 1つ前の標本を保持

def sample_mixed_gauss(mu, sigma, rate):
    '''
    GMMのサンプリング

    Parameters
    -----
    - mu: list
        期待値
    - sigma: list
        分散/標準偏差
    - rate: list
        混合率

    Notes
    -----
    - 単峰ガウス分布を混合率と一様分布乱数で制御することにより、混合ガウス分布をサンプリングする.
    '''
    dim = 'solo' if type(mu[0]) == int else 'multi'     # dimについて、'solo':1次元, 'multi':多次元
    if round(sum(rate)) != 1:
        raise Exception('混合率の和が1でありません.')
    else:
        for i in range(sample_size):
            u = np.random.rand()
            sum_ = 0
            for i in range(len(rate)):
                if sum_ < u < rate[i]+sum_:
                # 2021.10.16: Notice: ↑内包表記で書こうと試みたが...
                # if [sum_ < u < rate[i]+sum_ for i in range(len(rate))].any():
                # 注1. anyは標準リストではサポートなし。 また、内包表記ではすべての要素を見るため効率が悪い。適当なインデックスをif文に検知させるのも難しそう。
                # 注2. 逆に、for文による内包表記の中にif文を組み込むのは、よくあるやり方。
                    if dim == 'solo':
                        sample_list.append(np.random.normal(mu[i], sigma[i]))
                    elif dim == 'multi':
                        sample_list.append(np.random.multivariate_normal(mu[i], sigma[i], 1).tolist()[0])
                    break
                sum_ += rate[i]

def sample_hmm(mu, sigma, rate, state):
    '''
    HMMのサンプリング

    Parameters
    -----
    - mu: list
        期待値
    - sigma: list
        分散/標準偏差
    - rate: list
        遷移行列
    - state: int
        状態のラベル

    Notes
    -----
    - state: int
    　　初期状態. ただし、関数内のローカル変数として定義.
    '''
    dim = 'solo' if type(mu[0]) == int or type(mu[0]) == float else 'multi'     # dimについて、'solo':1次元, 'multi':多次元
    state_list = []     # テストコード用: 状態遷移列 (状態遷移の履歴) を保存
    if False in [len(rate) == len(rate[i]) for i in range(len(rate))]:
        raise Exception('与えられた遷移行列が正方行列でありません.')
    elif False in [round(sum(rate[i])) == 1 for i in range(len(rate))]:
        raise Exception('状態aからの遷移確率の和が1でありません.')
    else:
        for i in range(sample_size):
            #random_ = np.random.rand()  # 一様分布乱数を出力.
            random_ = 3/10                            # テストパターン1: 初期状態が0で、状態遷移しない。
            #random_ = 2/10 if i % 2 == 0 else 5/10    # テストパターン2: 初期状態が0で、状態0と1が交互に入れ換わる。
            #random_ = 8/10                            # テストパターン3: 初期状態が1で、状態遷移しない。
            #random_ = 5/10 if i % 2 == 0 else 2/10    # テストパターン4: 初期状態が1で、状態0と1が交互に入れ換わる。
            # HACK: 2021.10.27 22:15頃: 2状態を仮定しているため、状態遷移はビット演算を用いて実現.
            # 三項演算子について、
            # if文  : 状態維持
            # - 一番始めは初期状態を維持するため、条件にi == 0をorで追加.
            # - rate[state][state-1] < random_ <= rate[state][state+1] の時、状態維持.
            # - 遷移行列のrate[0][0], rate[len(rate)-1][len(rate)-1]の時、各々場合分けして処理.→ 　各々0, 1を出力.
            # else文: 状態遷移
            # - ↑それ以外の時、状態遷移.
            #state = state if ((i == 0) or   \  # 三項演算子について、if文内の条件が!=. ↓下記は==.
            #                  ((rate[state][state-1] if state != 0 else 0) < random_ <=  \
            #                        (rate[state][state+1] if state != len(rate)-1 else 1)))   \
            #              else int(format(~state & 0x1, '01b'))
            state = state if ((i == 0) or   \
                              ((0 if state == 0 else rate[state][state]) < random_ <=  \
                                    (1 if state == len(rate)-1 else rate[state][state])))   \
                          else int(format(~state & 0x1, '01b'))
            state_list.append(state)    # テストコード用: i回目での状態を追加.
            if dim == 'solo':
                sample_list.append(np.random.normal(mu[state], sigma[state]))
            elif dim == 'multi':
                sample_list.append(np.random.multivariate_normal(mu[state], sigma[state], 1).tolist()[0])
            else:
                raise Exception('関数内ローカル変数dimの設定が正しくありません.')
    return state_list   # テストコード用: 状態遷移の履歴を出力.


def main():
    #config_ = config.Conf()     # オブジェクトを生成 (↓ヒストグラムのプロットまでの処理は全てコメントアウト.)
    # 1. 所望の分布
    #p = lambda theta: gauss.gauss(theta, mu=0, sigma=1)       # 単変量ガウス分布
    p = lambda theta: gauss.multidim_gauss(theta,   \
                                            mu = np.matrix([0]).T,  \
                                            sigma = np.matrix([1]), \
                                          )                                         # 多変量ガウス分布
    #p = lambda theta: gauss.mixed_gauss(theta,  \
    #                                    (gauss.gauss(theta, mu=0, sigma=1), 1/4),   \
    #                                    (gauss.gauss(theta, mu=5, sigma=1), 1/4),   \
    #                                    (gauss.gauss(theta, mu=3, sigma=1), 2/4))   # 単変量混合ガウス分布
    # 2. 標本列の生成
    #metropolis(p)                                          # メトロポリス法
    #multidim_metropolis(p)                                  # 多変量メトロポリス法
    #metropolis_hastings(p)                                 # メトロポリス・ヘイスティングス法
    #sig = [[1 if i == j else 0 for i in range(6)] for j in range(6)]    # GMMのサンプリング1
    #sample_mixed_gauss(mu = [[0, 0, 0, 0, 0, 0], [5, 5, 5, 5, 5, 5]],
    #                   sigma = [sig, sig],
    #                   rate = [1/2, 1/2],
    #                  )
    #sample_mixed_gauss(mu = [0, 5],                         # GMMのサンプリング2
    #                   sigma = [1, 1],
    #                   rate = [1/2, 1/2],
    #                  )
    #sig = [[1, 0], [0, 1]]                                  # 多変量GMMのサンプリング1
    #sample_multidim_mixed_gauss(mu = [[0, 0], [3, 3], [6, 6], [9, 9], [12, 12]],
    #                            sigma = [sig, sig, sig, sig, sig],
    #                            rate = [1/6, 1/6, 2/6, 1/6, 1/6],
    #                           )
    #sig = [[1, 0], [0, 1]]                                  # 多変量GMMのサンプリング2
    #sample_multidim_mixed_gauss(mu = [[0, 0], [3, 3]],
    #                            sigma = [sig, sig],
    #                            rate = [1/2, 1/2],
    #                           )
    sample_hmm(mu = [0, 10],                                # HMMのサンプリング
               sigma = [1, 1],
               rate = [[9/10, 1/10], [1/10, 9/10]],    # テストパターン1, 3に類似
               #rate = [[1/10, 9/10], [9/10, 1/10]],    # テストパターン2, 4に類似
               state = 1
              )
    #print(sample_list[-10:])
    print(sample_list[:10])
    ### 3. 標本列のヒストグラム
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ## 1次元
    ##ax = plt.hist(sample_list, bins=100)
    ## 2次元
    ##x, y = sample_list                         # エラー
    ##x, y = np.vstack(sample_list)              # エラー
    #x, y = np.vstack(sample_list, sample_list)  # エラー
    #H = ax.hist2d(x, y, bins=[np.linspace(-30,30,61),np.linspace(-30,30,61)], cmap=cm.jet)
    #fig.colorbar(H[3],ax=ax)
    #plt.show()

if __name__ == '__main__':
    main()
