import numpy as np
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Loss Plot')
parser.add_argument('--file-name', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--p', default=0.0, type=float,
                    help='p')
parser.add_argument('--block-size', default=50, type=int)
args = parser.parse_args()

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
row_permute_sort_index = np.load('./synthetic_real_data_v4/row_permute_sort_index_B{}_noise{}_p{}.npy'.format(Block_size, N_noise, p_))


L_best =  np.load('./synthetic_real_data_v4_{}/pred_npy_{}.npy'.format(args.file_name, 10000))
best_val = 1000.0
best_iteration = 0

for iteration in range(1000, 50000, 1000):
    L = np.load('./synthetic_real_data_v4_{}/pred_npy_{}.npy'.format(args.file_name, iteration))
    M_diff = M - L
    M_diff_val = M_diff.ravel()[dropped_idx_val]

    residual_val = ((M_diff_val)**2).mean()
    if residual_val < best_val:
        best_val = residual_val * 1.0
        L_best = L.copy()
        best_iteration = iteration
    # print(((M_diff_val) ** 2).mean())

M_diff = M - L_best
M_diff_test = M_diff.ravel()[dropped_idx_test]
test_mse = ((M_diff_test)**2).mean()
print('best val mse: ', best_val)
print('test mse: ', test_mse)
print('best iteration: ', best_iteration)


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
        im1 = ax1.matshow(M_diff[:60, :60])
        ax1.set_title('recovered S, \n Test MSE={:0.5f}, \n Recover Error={}, \n Recover Error (Block)={}'.format(test_mse,
                                                                                                                  recovery_error,
                                                                                                                  recovery_error_block), fontsize=15, pad=30)

        im2 = ax2.matshow(S_thres[:60, :60])
        ax2.set_title('original S'.format(10), fontsize=15, pad=30)

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        plt.savefig(
            './results_dmf_realdata_v4/heatmap_dmf_iter{}_B{}_noise{}_p{}_threshold{}.png'.format(best_iteration, Block_size, N_noise, p_, threshold),
            bbox_inches='tight')

print('threshold_best: ', threshold_best)

