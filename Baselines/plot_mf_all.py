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
from numpy import linalg as LA

parser = argparse.ArgumentParser(description='MF params')
parser.add_argument('--file-dir', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--file-name', default='', type=str, metavar='PATH',
                    help='path to load file')
args = parser.parse_args()

S_gt = np.load('./{}/S_{}.npy'.format(args.file_dir, args.file_name))
M = np.load('./{}/M_{}.npy'.format(args.file_dir, args.file_name))
not_nan_idx = np.load('./{}/not_nan_idx_{}.npy'.format(args.file_dir, args.file_name))
dropped_idx = np.load('./{}/dropped_idx_{}.npy'.format(args.file_dir, args.file_name))
dropped_idx_val = np.load('./{}/dropped_idx_val_{}.npy'.format(args.file_dir, args.file_name))
dropped_idx_test = np.load('./{}/dropped_idx_test_{}.npy'.format(args.file_dir, args.file_name))
block_index = np.load('./{}/block_idx_test_{}.npy'.format(args.file_dir, args.file_name))
row_permute_sort_index = np.load('./{}/row_permute_sort_index_{}.npy'.format(args.file_dir, args.file_name))
col_permute_sort_index = np.load('./{}/col_permute_sort_index_{}.npy'.format(args.file_dir, args.file_name))

L_best = np.load(
    './{}/results_cvx_mf_{}/L_hat_cvx_mf_{}_eps{}.npy'.format(args.file_dir, args.file_name, args.file_name, 0.01))

best_val = 1000.0
best_eps = 0.0
for EPS in [0.01, 0.05, 0.1, 0.2, 0.3]:
    L = np.load(
        './{}/results_cvx_mf_{}/L_hat_cvx_mf_{}_eps{}.npy'.format(args.file_dir, args.file_name, args.file_name, EPS))
    M_diff = M - L
    M_diff_val = M_diff.ravel()[dropped_idx_val]

    residual_val = (M_diff_val ** 2).mean()
    # if residual_val < best_val:
    best_val = residual_val * 1.0
    L_best = L.copy()
    best_eps = EPS * 1.0
    print((M_diff_val ** 2).mean())

    M_diff = M - L_best
    M_diff_test = M_diff.ravel()[dropped_idx_test]
    test_mse = (M_diff_test ** 2).mean()
    print('test mse:', test_mse)

    recovery_error_best = 1000000.0
    threshold_best = 0.0

    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
        M_diff = M - L_best
        M_diff[np.where(np.abs(M_diff) < threshold)] = 0
        M_diff.ravel()[dropped_idx] = 0
        M_diff = M_diff[row_permute_sort_index, :]
        M_diff = M_diff[:, col_permute_sort_index]

        S_thres = S_gt.copy()
        S_thres[np.where(np.abs(S_thres) < threshold)] = 0
        S_thres.ravel()[dropped_idx] = 0
        S_thres = S_thres[row_permute_sort_index, :]
        S_thres = S_thres[:, col_permute_sort_index]

        M_diff_binary = M_diff.copy()
        M_diff_binary[np.where(M_diff < 0.0)] = 0
        M_diff_binary[np.where(M_diff > 0.0)] = 1.0
        S_thres_binary = S_thres.copy()
        S_thres_binary[np.where(S_thres < 0.0)] = 0
        S_thres_binary[np.where(S_thres > 0.0)] = 1.0

        recovery_error = np.sum(np.abs(M_diff_binary.ravel() - S_thres_binary.ravel()))
        recovery_error_block = np.sum(np.abs(M_diff_binary.ravel()[block_index] - S_thres_binary.ravel()[block_index]))
        print('recovery_error: ', recovery_error)
        print('recovery_error_block: ', recovery_error_block)

        # if recovery_error < recovery_error_best:
        recovery_error_best = recovery_error * 1.0
        threshold_best = threshold

        fig, axes = plt.subplots(ncols=1, figsize=(5, 4))
        ax1 = axes

        im1 = ax1.matshow(M_diff[:60, :60])
        ax1.set_title(
            'recovered S, \n Test MSE={:0.5f}, \n Recover Error={}, \n Recover Error (Block)={}'.format(test_mse,
                                                                                                        recovery_error,
                                                                                                        recovery_error_block),
            fontsize=15, pad=30)

        fig.colorbar(im1, ax=ax1)
        plt.savefig('./{}/results_cvx_mf_{}/single_heatmap_cvx_mf_eps{}_{}_threshold{}.png'.format(args.file_dir,
                                                                                            args.file_name,
                                                                                            best_eps,
                                                                                            args.file_name,
                                                                                            threshold),
                    bbox_inches='tight')

        # fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
        # ax1, ax2 = axes
        #
        # im1 = ax1.matshow(M_diff[:60, :60])
        # ax1.set_title(
        #     'recovered S, \n Test MSE={:0.5f}, \n Recover Error={}, \n Recover Error (Block)={}'.format(test_mse,
        #                                                                                                 recovery_error,
        #                                                                                                 recovery_error_block),
        #     fontsize=15, pad=30)
        #
        # im2 = ax2.matshow(S_thres[:60, :60])
        # ax2.set_title('original S'.format(10), fontsize=15, pad=30)
        #
        # fig.colorbar(im1, ax=ax1)
        # fig.colorbar(im2, ax=ax2)
        # plt.savefig('./{}/results_cvx_mf_{}/heatmap_cvx_mf_eps{}_{}_threshold{}.png'.format(args.file_dir,
        #                                                                                     args.file_name,
        #                                                                                     best_eps,
        #                                                                                     args.file_name,
        #                                                                                     threshold),
        #             bbox_inches='tight')

    print('threshold_best: ', threshold_best)
