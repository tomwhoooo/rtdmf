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
parser.add_argument('--TAU', default=0.1, type=float,
                    help='parameter for RPCA (default: 1.0)')
parser.add_argument('--p', default=0.0, type=float,
                    help='p')
args = parser.parse_args()

TAU = args.TAU
Block_size = args.block_size
p_ = args.p
N_noise = 0.05

S_gt = np.load('./synthetic_real_data_v4/S_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
M = np.load('./synthetic_real_data_v4/M_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
not_nan_idx = np.load('./synthetic_real_data_v4/not_nan_idx_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
dropped_idx = np.load('./synthetic_real_data_v4/dropped_idx_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
dropped_idx_val = np.load('./synthetic_real_data_v4/dropped_idx_val_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
dropped_idx_test = np.load('./synthetic_real_data_v4/dropped_idx_test_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
block_index = np.load('./synthetic_real_data_v4/block_idx_test_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))
row_permute_sort_index = np.load(
    './synthetic_real_data_v4/row_permute_sort_index_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))

Mask = np.zeros_like(M)
non_dropped_idx = np.setdiff1d(not_nan_idx, dropped_idx)
Mask.ravel()[non_dropped_idx] = 1

L_best = np.load("./cvx_rpca_realdata_v4/L_hat_cvx_rpca_B{}_noise{}_p{}_tau{}.npy".format(Block_size, N_noise, p_, 0.1))
best_val = 1000.0


for TAU in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    L = np.load("./cvx_rpca_realdata_v4/L_hat_cvx_rpca_B{}_noise{}_p{}_tau{}.npy".format(Block_size, N_noise, p_, TAU))
    M_diff = M - L
    M_diff_val = M_diff.ravel()[dropped_idx_val]

    residual_val = ((M_diff_val)**2).mean()
    if residual_val < best_val:
        best_val = residual_val * 1.0
        L_best = L.copy()
    print(((M_diff_val) ** 2).mean())
print('best val mse: ', best_val)
M_diff = M - L_best
M_diff_test = M_diff.ravel()[dropped_idx_test]
test_mse = ((M_diff_test)**2).mean()
print('test mse:', test_mse)

# for TAU in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
#     L = np.load("./cvx_rpca_realdata_v4/L_hat_cvx_rpca_B{}_noise{}_p{}_tau{}.npy".format(Block_size, N_noise, p_, TAU))
#     L_best = L.copy()
#     M_diff = M - L

recovery_error_best = 1000000.0
threshold_best = 0.0

for threshold in [0.05, 0.1, 0.15, 0.2]:
    M_diff = M - L_best
    M_diff[np.where(np.abs(M_diff) < threshold)] = 0
    M_diff.ravel()[dropped_idx] = 0
    M_diff = M_diff[row_permute_sort_index, :]
    S_thres = S_gt.copy()
    S_thres[np.where(np.abs(S_thres) < threshold)] = 0
    S_thres.ravel()[dropped_idx] = 0
    S_thres = S_thres[row_permute_sort_index, :]

    M_diff_binary = M_diff.copy()
    M_diff_binary[np.where(M_diff < 0.0)] = 0
    M_diff_binary[np.where(M_diff > 0.0)] = 1.0
    S_thres_binary = S_thres.copy()
    S_thres_binary[np.where(S_thres < 0.0)] = 0
    S_thres_binary[np.where(S_thres > 0.0)] = 1.0
    recovery_error = np.sum(np.abs(M_diff_binary.ravel() - S_thres_binary.ravel()))
    recovery_error_block = np.sum(np.abs(M_diff_binary.ravel()[block_index] - S_thres_binary.ravel()[block_index]))
    print('recovery_error: ', recovery_error)

    if recovery_error < recovery_error_best:
        recovery_error_best = recovery_error * 1.0
        threshold_best = threshold
        fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
        ax1, ax2 = axes
        # max_value = np.maximum(v.max(), M_diff[:, :2].max())
        print(M_diff[1, :10])
        im1 = ax1.matshow(M_diff[:60, :60])
        ax1.set_title('recovered S, \n Test MSE={:0.5f}, \n Recover Error={}, \n Recover Error (Block)={}'.format(test_mse,
                                                                                                                  recovery_error,
                                                                                                                  recovery_error_block), fontsize=15, pad=30)

        im2 = ax2.matshow(S_thres[:60, :60])
        ax2.set_title('original S'.format(10), fontsize=15, pad=30)

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        # plt.savefig(
        #     './cvx_rpca_realdata_v4/heatmap_cvx_rpca_B{}_noise{}_p{}_threshold{}_tau{}.png'.format(Block_size, N_noise, p_, threshold, TAU),
        #     bbox_inches='tight')
        plt.savefig(
            './cvx_rpca_realdata_v4/heatmap_cvx_rpca_B{}_noise{}_p{}_threshold{}.png'.format(Block_size, N_noise, p_, threshold),
            bbox_inches='tight')

print('threshold_best: ', threshold_best)

# L_best = np.load("./cvx_rpca_realdata_v4/L_hat_cvx_rpca_B{}_noise{}_p{}_tau{}.npy".format(Block_size, N_noise, p_, 0.1))
# best_val = 1000.0
#
# for TAU in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
#     L = np.load("./cvx_rpca_realdata_v4/L_hat_cvx_rpca_B{}_noise{}_p{}_tau{}.npy".format(Block_size, N_noise, p_, TAU))
#     M_diff = M - L
#     M_diff_val = M_diff.ravel()[dropped_idx_val]
#
#     residual_val = ((M_diff_val) ** 2).mean()
#     # if residual_val < best_val:
#     #     best_val = residual_val * 1.0
#     #     L_best = L.copy()
#     print(((M_diff_val) ** 2).mean())
#     print('val mse: ', residual_val)
#     M_diff = M - L
#     M_diff_test = M_diff.ravel()[dropped_idx_test]
#     test_mse = ((M_diff_test) ** 2).mean()
#     print('test mse:', test_mse)
#
#     recovery_error_best = 1000000.0
#     threshold_best = 0.0
#
#     for threshold in [0.05, 0.1, 0.15, 0.2]:
#         M_diff = M - L
#         M_diff[np.where(np.abs(M_diff) < threshold)] = 0
#         M_diff.ravel()[dropped_idx] = 0
#         M_diff = M_diff[row_permute_sort_index, :]
#         S_thres = S_gt.copy()
#         S_thres[np.where(np.abs(S_thres) < threshold)] = 0
#         S_thres.ravel()[dropped_idx] = 0
#         S_thres = S_thres[row_permute_sort_index, :]
#
#         M_diff_binary = M_diff.copy()
#         M_diff_binary[np.where(M_diff < 0.0)] = 0
#         M_diff_binary[np.where(M_diff > 0.0)] = 1.0
#         S_thres_binary = S_thres.copy()
#         S_thres_binary[np.where(S_thres < 0.0)] = 0
#         S_thres_binary[np.where(S_thres > 0.0)] = 1.0
#         recovery_error = np.sum(np.abs(M_diff_binary.ravel() - S_thres_binary.ravel()))
#         recovery_error_block = np.sum(np.abs(M_diff_binary.ravel()[block_index] - S_thres_binary.ravel()[block_index]))
#         print('recovery_error: ', recovery_error)
#
#         if recovery_error < recovery_error_best:
#             recovery_error_best = recovery_error * 1.0
#             threshold_best = threshold
#             fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
#             ax1, ax2 = axes
#             # max_value = np.maximum(v.max(), M_diff[:, :2].max())
#             im1 = ax1.matshow(M_diff[:5 * Block_size, :5 * Block_size])
#             ax1.set_title(
#                 'recovered S (TAU={}), \n Test MSE={:0.5f}, \n Recover Error={}, \n Recover Error (Block)={}'.format(
#                     TAU,
#                     test_mse,
#                     recovery_error,
#                     recovery_error_block), fontsize=15, pad=30)
#
#             im2 = ax2.matshow(S_thres[:5 * Block_size, :5 * Block_size])
#             ax2.set_title('original S'.format(10), fontsize=15, pad=30)
#
#             fig.colorbar(im1, ax=ax1)
#             fig.colorbar(im2, ax=ax2)
#             plt.savefig(
#                 './cvx_rpca_realdata_v4/heatmap_cvx_rpca_TAU{}_B{}_noise{}_p{}_threshold{}.png'.format(TAU,
#                                                                                               Block_size, N_noise, p_,
#                                                                                               threshold),
#                 bbox_inches='tight')
#
#     print('threshold_best: ', threshold_best)
