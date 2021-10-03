import random
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('.')
import gauss


def main():
    '''
    メトロポリス法を実装.
    メトロポリス・ヘイスティングス法の簡易版.
    代理分布: Normal(sample_list[i], 1)
    '''
    mu = 0
    sigma = 1
    p = lambda theta: gauss.gauss(theta, mu=mu, sigma=sigma)  # 所望の分布

    sample_list = []
    sample_list.append(10) # 初期値を適当に10と定めた

    for i in range(100000):
        theta = np.random.normal(sample_list[i], 1) # 提案値
        a = min(1, p(theta) / p(sample_list[i]))

        u = np.random.rand() # 0 ~ 1 の一様乱数を生成

        if u < a: # 受諾
            sample_list.append(theta)
        else: # 拒否
            sample_list.append(sample_list[i])  # 1つ前の標本を保持

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plt.hist(sample_list, bins=100)
    plt.show()

if __name__ == '__main__':
    main()
