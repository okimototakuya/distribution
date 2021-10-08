import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


sample_size = 100   # サンプルサイズ

def gauss(x, mu, sigma):
    '''
    (単峰)ガウス分布を、np.ndarray型で返す.

    Parameters
    -----
    x: 確率変数値 (int, numpy.ndarray, numpy.matrix)
    mu: 期待値スカラー (int)
    sigma: 標準偏差スカラー (int)   注. sigmaは分散でなく、標準偏差であることに注意すること. 

    Note
    -----
    - numpy=1.20.1/numpy-base=1.20.1の環境において、multidim_gaussのパラメータを1次元にした場合、返り値が一致した.
    - 2021.10.6 昼頃: ↑ただし、標準正規分布の場合について。
    - 2021.10.6 夜: ↑標準正規分布以外で、値がほぼ一致することを確認した。おそらく数値誤差のよる差分。
    '''
    gauss_y = np.exp(-((x-mu)**2) / (2*(sigma**2)))     \
                    / (np.sqrt(2*np.pi) * sigma)     # 上行:密度関数の本体, 下行:規格化定数
    return gauss_y

def multidim_gauss(x, mu, sigma):
    '''
    多変量ガウス分布を、np.ndarray型で返す.

    Parameters
    -----
    x: 確率変数値 (int, numpy.ndarray, numpy.matrix)
    mu: 期待値ベクトル (numpy.matrix配列の転置)
    sigma: 共分散分散行列 (numpy.matrix行列)            # 1変量分布の場合、標準偏差でなく分散を与えること.

    Notes
    -----
    - [参考]: https://qiita.com/g-k/items/698c7f9e4a213d73197b
    '''
    d = x.T.ndim
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)      # np.matrix型
    matrix_z = np.exp(-(x - mu).T@inv@(x - mu)/2.0)    \
               / (np.sqrt((2*np.pi)**d * det))     # 引数をnp.matrix型で与えた場合, 返り値もnp.matrix型.
    return np.diag(matrix_z)     # 対角成分の抽出. → 2021/10/6: ToDo: 何故これでプロットできるようになったのか不明.

def mixed_gauss(x, *input_gauss):
    '''
    混合ガウス分布を、np.ndarray型で返す.

    Parameters
    -----
    x: 2021.10.8[HACK]: input_gaussを引数にしており、不要？
    input_gauss: ガウス分布の密度関数 (np.ndarray型) とその混合率のタプルを要素に持つタプル
    '''
    if sum(input_gauss[i][1] for i in range(len(input_gauss))) != 1:
        raise Exception('混合率の和が1ではありません。')
    else:
        # input_gauss[i][0]: ガウス分布の密度関数 (np.ndarray型), input_gauss[i][1]: 混合率
        mixed_gauss_y = sum(input_gauss[i][0] * input_gauss[i][1] for i in range(len(input_gauss)))
        return mixed_gauss_y

def main():
    '''
    分布をプロットする.
    '''
    mu = 0                     # 通った
    sigma = 1
    #mu = [0, 0]                # 通った
    #sigma = [[2, 1], [1, 3]]
    #mu = [0]                    # 通らない
    #sigma = [1]                # 2021/10/7: FIXME: テストスクリプトより、分布の定義までは正常に動作するが、プロットについては１次元の方法をとらなければいけない。
    #if type(mu)=='int' or len(mu) == 1:
    if type(mu)==int:
        # 1. 座標軸の設定
        x = np.linspace(-3, 3, sample_size)
        # 2. ガウス分布の密度関数
        gauss_y = gauss(x, mu=mu, sigma=np.sqrt(sigma))                             # 単変量ガウス分布 (gauss.gauss)
        #gauss_y = multidim_gauss(x.T, mu=np.matrix([0]).T, sigma=np.matrix([5]))   # (一般に)多変量ガウス分布 (gauss.multidim_gauss)
        gauss_y_a = gauss(x, mu=-2, sigma=np.sqrt(2))
        gauss_y_b = gauss(x, mu=2, sigma=np.sqrt(3/2))
        mixed_gauss_y = mixed_gauss(x, (gauss_y_a, 3/4), (gauss_y_b, 1/4))   # (単変量)混合ガウス分布 (gauss.mixed_gauss)
        # 3. ガウス分布のプロット
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, mixed_gauss_y)
    #elif len(mu) == 2:
    elif type(mu)==list:
        # 1. 座標軸の設定
        x = y = np.linspace(-3, 3, sample_size)
        X, Y = np.meshgrid(x, y)
        z = np.c_[X.ravel(),Y.ravel()]
        # 2. ガウス分布の密度関数
        Z = multidim_gauss(z.T, mu=np.matrix(mu).T, sigma=np.matrix(sigma))  # 多変量ガウス分布 (gauss.multidim_gauss)
        # 3. ガウス分布のプロット
        shape = X.shape
        Z = Z.reshape(shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    #elif len(mu) == 3:
    elif type(mu) == float:
        pass
    else:
        raise Exception('プロットするのに、分布の次元数が不適切です.')
    plt.show()


if __name__ == '__main__':
    main()
