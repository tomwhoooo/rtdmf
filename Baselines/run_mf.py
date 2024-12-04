import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_low_rank_matrix
from matplotlib import pyplot as plt
import cvxpy as cvx
import numpy as np
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='RPCA params')
parser.add_argument('--block-size', default=50, type=int,
                    help='block size')
parser.add_argument('--p', default=0.05, type=float,
                    help='p')
parser.add_argument('--EPS', default=0.1, type=float,
                    help='parameter for RPCA (default: 1.0)')
args = parser.parse_args()

EPS = args.EPS
Block_size = args.block_size
p_ = args.p
N_noise = 0.05

S_gt = np.load('./synthetic_real_data_v4/S_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
M = np.load('./synthetic_real_data_v4/M_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
not_nan_idx = np.load('./synthetic_real_data_v4/not_nan_idx_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
dropped_idx = np.load('./synthetic_real_data_v4/dropped_idx_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
row_permute_sort_index = np.load('./synthetic_real_data_v4/row_permute_sort_index_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))


Mask = np.zeros_like(M)
non_dropped_idx = np.setdiff1d(not_nan_idx, dropped_idx)
Mask.ravel()[non_dropped_idx] = 1


def get_cvx_opt_constraints(L, shape, M, non_dropped_idx):
    A = np.zeros_like(M)
    mask = np.zeros_like(M)
    A.ravel()[non_dropped_idx] = M.ravel()[non_dropped_idx]
    mask.ravel()[non_dropped_idx] = 1
    eps = EPS
    constraints = [cvx.abs(cvx.multiply(L - A, mask)) <= eps]
    return constraints

shape = M.shape
L = cvx.Variable(shape=shape)
objective = cvx.Minimize(cvx.norm(L, 'nuc'))
constraints = get_cvx_opt_constraints(L, shape, M, non_dropped_idx)

problem = cvx.Problem(objective, constraints)
problem.solve(verbose=True, use_indirect=False)
print("Optimal value: ", problem.value)
print("L:\n", L.value)

np.save('./cvx_mf_realdata_v4/L_hat_cvx_mf_B{}_noise{}_p{}_eps{}.npy'.format(Block_size, N_noise, p_, EPS), L.value)
