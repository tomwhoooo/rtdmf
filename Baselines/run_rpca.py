import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_low_rank_matrix
from matplotlib import pyplot as plt
import cvxpy as cvx
import numpy as np
import argparse
import numpy as np
import pandas as pd
import os


parser = argparse.ArgumentParser(description='RPCA params')
parser.add_argument('--TAU', default=1.0, type=float,
                    help='parameter for RPCA (default: 1.0)')
parser.add_argument('--file-dir', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--file-name', default='', type=str, metavar='PATH',
                    help='path to load file')
args = parser.parse_args()


TAU = args.TAU

S_gt = np.load('./{}/S_{}.npy'.format(args.file_dir, args.file_name))
M = np.load('./{}/M_{}.npy'.format(args.file_dir, args.file_name))
not_nan_idx = np.load('./{}/not_nan_idx_{}.npy'.format(args.file_dir, args.file_name))
dropped_idx = np.load('./{}/dropped_idx_{}.npy'.format(args.file_dir, args.file_name))
row_permute_sort_index = np.load('./{}/row_permute_sort_index_{}.npy'.format(args.file_dir, args.file_name))


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

shape = M.shape
L = cvx.Variable(shape=shape)
S = cvx.Variable(shape=shape)
tau = TAU / np.sqrt(max(shape))
objective = cvx.Minimize(cvx.norm(L, 'nuc') + tau * cvx.sum(cvx.sum(cvx.abs(S))))
constraints = get_cvx_opt_constraints_rpca(L, S, shape, M, non_dropped_idx)

problem = cvx.Problem(objective, constraints)
problem.solve(verbose=True, use_indirect=False)
print("Optimal value: ", problem.value)

outdir = './{}/results_cvx_rpca_{}'.format(args.file_dir, args.file_name)
print('output directory: ', outdir)
if not os.path.exists(outdir):
    os.makedirs(outdir)

np.save('{}/L_hat_cvx_rpca_{}_tau{}.npy'.format(outdir, args.file_name, TAU), L.value)
np.save('{}/S_hat_cvx_rpca_{}_tau{}.npy'.format(outdir, args.file_name, TAU), S.value)
