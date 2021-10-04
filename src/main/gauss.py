import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


sample_size = 100   # サンプルサイズ

def gauss(x, mu, sigma):
    '''
    (単峰)ガウス分布を、np.ndarray型で返す.
    '''
    gauss_y = np.exp(-(x-mu)**2 / 2*sigma**2)     \
                    / np.sqrt(2 * np.pi * sigma**2)     # 上行:密度関数の本体, 下行:規格化定数
    return gauss_y

def multidim_gauss(x, mu, sigma):
    '''
    '''
    d = x.ndim
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    print(det)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = x.ndim
    print(inv)
    return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))    # [注]. 横ベクトルがデフォルト

def mixed_gauss(x, input_gauss):
    '''
    混合ガウス分布を、np.ndarray型で返す.
    '''
    mixed_gauss_y = np.array(input_gauss)
    return mixed_gauss_y

def main():
    # 1. 座標軸の設定
    #x = np.linspace(-3, 3, sample_size)        # 単変量
    x = y = np.linspace(-3, 3, sample_size)     # ２変量
    X, Y = np.meshgrid(x, y)
    z = np.c_[X.ravel(),Y.ravel()]
    # 2. ガウス分布の密度関数
    #gauss_y = gauss(x, mu=0, sigma=1)           # (単変量)ガウス分布
    gauss_Z = multidim_gauss(z, mu=np.array([0, 0]), sigma=np.array([[1, 0], [0, 1]]))  # 多変量ガウス分布
    #gauss_y = mixed_gauss(x, gauss_y)     # (単変量)混合ガウス分布
    # 3. ガウス分布のプロット
    shape = X.shape
    gauss_Z = gauss_Z.reshape(shape)
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(x, gauss_y)
    ax.plot_surface(X, Y, gauss_Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.show()


if __name__ == '__main__':
    main()
