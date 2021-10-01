import random
import matplotlib.pyplot as plt
import numpy as np


sample_size = 100

def gauss(x, mu, sigma):
    gauss_y = np.exp(-(x-mu)**2 / 2*sigma**2)     \
                    / np.sqrt(2 * np.pi * sigma**2)     # 上行:密度関数の本体, 下行:規格化定数
    return gauss_y

def main():
    sigma = 1   # 標準偏差
    mu = 0      # 平均値
    #x = [random.gauss(mu=0, sigma=1) for _ in range(sample_size)]
    x = np.linspace(-3, 3, sample_size)
    gauss_y = gauss(x, mu=0, sigma=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, gauss_y)
    plt.show()


if __name__ == '__main__':
    main()
