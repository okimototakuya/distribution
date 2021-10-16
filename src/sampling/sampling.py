import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import config
sys.path.append('../distribution')
import gauss

sample_list = []
sample_list.append(10) # 初期値を適当に10と定めた

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
    for i in range(100000):
        theta = np.random.normal(sample_list[i], 1) # 提案値
        a = min(1, p(theta) / p(sample_list[i]))

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

    for i in range(100000):
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

    Notes
    -----
    - 単峰ガウス分布を混合率で制御することにより、混合ガウス分布をサンプリングする.
    '''
    if sum(rate) != 1:
        raise Exception('混合率の和が1でありません.')
    else:
        for i in range(100000):
            u = np.random.rand()
            if u < rate[0]:
                sample_list.append(np.random.normal(mu[0], sigma[0]))
            elif rate[0] <= u < rate[1]+rate[0]:
                sample_list.append(np.random.normal(mu[1], sigma[1]))
            else:
                sample_list.append(np.random.normal(mu[2], sigma[2]))


def main():
    #config_ = config.Conf()     # オブジェクトを生成 (↓ヒストグラムのプロットまでの処理は全てコメントアウト.)
    # 1. 所望の分布
    #mu = 0                                                         # 単変量ガウス分布
    #sigma = 1
    #p = lambda theta: gauss.gauss(theta, mu=mu, sigma=sigma)       # theta : intまたはnp.ndarray
    p = lambda theta: gauss.mixed_gauss(theta,  \
                                        (gauss.gauss(theta, mu=0, sigma=1), 1/4),   \
                                        (gauss.gauss(theta, mu=5, sigma=1), 1/4),   \
                                        (gauss.gauss(theta, mu=3, sigma=1), 2/4))   # 単変量混合ガウス分布
    # 2. 標本列の生成
    #metropolis(p)                                                                  # メトロポリス法
    #metropolis_hastings(p)                                                         # メトロポリス・ヘイスティングス法
    sample_mixed_gauss(mu = [0, 3, 6], sigma = [1, 1, 1], rate = [2/4, 1/4, 1/4])   # GMMのサンプリング
    # 3. 標本列のヒストグラム
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plt.hist(sample_list, bins=100)
    plt.show()

if __name__ == '__main__':
    main()
