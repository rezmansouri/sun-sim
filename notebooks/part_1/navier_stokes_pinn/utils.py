from scipy.io import loadmat
import torch
import numpy as np


def get_data(path, N_train):
    data = loadmat(path)

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    x_test = X_star[:, 0:1]
    y_test = X_star[:, 1:2]
    p_test = P_star[:, 0:1]
    u_test = U_star[:, 0:1, 0]
    v_test = U_star[:, 1:2, 0]
    t_test = np.ones((x_test.shape[0], x_test.shape[1]))

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    # Training Data
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]
    
    return x_train, y_train, t_train, u_train, v_train, x_test, y_test, t_test, u_test, v_test, p_test
