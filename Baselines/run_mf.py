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
import os


parser = argparse.ArgumentParser(description='MF params')
parser.add_argument('--EPS', default=0.1, type=float,
                    help='parameter for RPCA (default: 1.0)')
parser.add_argument('--file-dir', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--file-name', default='', type=str, metavar='PATH',
                    help='path to load file')
args = parser.parse_args()

EPS = args.EPS

S_gt = np.load('./{}/S_{}.npy'.format(args.file_dir, args.file_name))
M = np.load('./{}/M_{}.npy'.format(args.file_dir, args.file_name))
not_nan_idx = np.load('./{}/not_nan_idx_{}.npy'.format(args.file_dir, args.file_name))
dropped_idx = np.load('./{}/dropped_idx_{}.npy'.format(args.file_dir, args.file_name))

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

outdir = './{}/results_cvx_mf_{}'.format(args.file_dir, args.file_name)
print('output directory: ', outdir)
if not os.path.exists(outdir):
    os.makedirs(outdir)

np.save('{}/L_hat_cvx_mf_{}_eps{}.npy'.format(outdir, args.file_name, EPS), L.value)
