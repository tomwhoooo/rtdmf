import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_low_rank_matrix
from matplotlib import pyplot as plt
import cvxpy as cvx
import numpy as np
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='RPCA params')
parser.add_argument('--block-size', default=50, type=int,
                    help='block size')
parser.add_argument('--p', default=0.05, type=float,
                    help='p')
parser.add_argument('--TAU', default=1.0, type=float,
                    help='parameter for RPCA (default: 1.0)')
args = parser.parse_args()


TAU = args.TAU
Block_size = args.block_size
N_noise = 0.05
p_ = args.p

S_gt = np.load('./synthetic_real_data_v4/S_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
M = np.load('./synthetic_real_data_v4/M_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
not_nan_idx = np.load('./synthetic_real_data_v4/not_nan_idx_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
dropped_idx = np.load('./synthetic_real_data_v4/dropped_idx_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
row_permute_sort_index = np.load('./synthetic_real_data_v4/row_permute_sort_index_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))


Mask = np.zeros_like(M)
non_dropped_idx = np.setdiff1d(not_nan_idx, dropped_idx)
Mask.ravel()[non_dropped_idx] = 1


def get_cvx_opt_constraints_rpca(L, S, shape, M, non_dropped_idx):
    A = np.zeros_like(M)
    mask = np.zeros_like(M)
    A.ravel()[non_dropped_idx] = M.ravel()[non_dropped_idx]
    mask.ravel()[non_dropped_idx] = 1
    eps = 1.e-5
    constraints = [cvx.abs(cvx.multiply(L + S - A, mask)) <= eps]
    return constraints

shape = (800, 100)
L = cvx.Variable(shape=shape)
S = cvx.Variable(shape=shape)
tau = TAU / np.sqrt(max(shape))
objective = cvx.Minimize(cvx.norm(L, 'nuc') + tau * cvx.sum(cvx.sum(cvx.abs(S))))
constraints = get_cvx_opt_constraints_rpca(L, S, shape, M, non_dropped_idx)

problem = cvx.Problem(objective, constraints)
problem.solve(verbose=True, use_indirect=False)
print("Optimal value: ", problem.value)

np.save('./cvx_rpca_realdata_v4/L_hat_cvx_rpca_B{}_noise{}_p{}_tau{}.npy'.format(Block_size, N_noise, p_, TAU), L.value)
np.save('./cvx_rpca_realdata_v4/L_hat_cvx_rpca_S{}_noise{}_p{}_tau{}.npy'.format(Block_size, N_noise, p_, TAU), S.value)
