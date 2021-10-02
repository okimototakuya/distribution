import random
import sys
import numpy as np
sys.path.append('.')
import gauss

def main():
    x = np.linspace(-3, 3, 100)
    mu = 0
    sigma = 1
    p_theta = lambda theta: gauss.gauss(theta, mu=mu, sigma=sigma)  # 所望の分布を取得


if __name__ == '__main__':
    main()
