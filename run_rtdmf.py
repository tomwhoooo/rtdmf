"""
This is an implementation of the RTDMF algorithm.
"""

import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np

from utils import init_logfile, log

# Argument parser

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--hidden-sizes', '--hs', nargs='+', type=int)
parser.add_argument('--file-dir', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--file-name', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--iterations', default=100000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch', default=0, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='learning rate', dest='lr')
parser.add_argument('--init-scale', default=0.001, type=float,
                    help='initial scale for network')
parser.add_argument('--initialization', default='gaussian', type=str, metavar='INIT',
                    help='initialization: gaussian, identity etc.')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPT',
                    help='optimizer')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--threshold-type', type=str, choices=['none', 'in-training', 'post-training'],
                    default='none', help='Type of thresholding to apply')
parser.add_argument('--threshold-k', type=float, default=0.3,
                    help='Maximum allowed deviation from original value')
parser.add_argument('--threshold-check-freq', type=int, default=10000,
                    help='Frequency of threshold checking for in-training-threshold')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
args = parser.parse_args()

# device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


class create_simulation(object):
    def __init__(self, params):
        self.simulation = params['simulation']
        if self.simulation:
            # generate ground truth matrix
            self.A = np.dot(np.random.rand(100, 5), np.random.rand(5, 100))
            B = self.A.copy()
            # save dropped indices
            ### random ###
            # self.dropped_idx = np.random.choice(B.size, 8000, replace=False)
            ### structured centering, i.e. most of the central columns will be missing ###
            ### subject to some noise ###
            col_idx = np.random.choice(100, 90, replace=False)
            row_idx = np.array(range(25, 75)) + np.floor(np.random.normal(0, 2, 50))
            self.dropped_idx = np.unique(np.array([row_idx * k for k in col_idx]).flatten().astype(int))
            # generate masked matrix
            B.ravel()[self.dropped_idx] = np.nan
            self.file = B


class param_saver(object):
    def __init__(self, params, cs=None):
        self.simulation = params['simulation']
        if self.simulation:
            self.A = cs.A
            self.dropped_idx = cs.dropped_idx
            self.file = cs.file
        if not self.simulation:
            # read in original matrix
            self.file_dir = params['file_dir']
            self.A = np.load('./synthetic_real_data_v4/M_{}.npy'.format(args.file_name))
            B = self.A.copy()
            print('matrix shape: ', B.shape)
            self.dropped_idx = np.load('./synthetic_real_data_v4/dropped_idx_{}.npy'.format(args.file_name))
            print('num of dropped index: ', len(self.dropped_idx))
            B.ravel()[self.dropped_idx] = np.nan
            self.file = B
        # insert the dimension into hidden sizes
        self.hidden_sizes = params['hidden_sizes']
        self.hidden_sizes.insert(0, self.file.shape[0])
        self.hidden_sizes.append(self.file.shape[1])
        self.n_iter = params['n_iter']
        self.lr = params['lr']
        self.optimizer = params['optimizer']
        self.initialization = params['initialization']


def get_e2e(model):
    weight = None
    for fc in model.children():
        assert isinstance(fc, nn.Linear) and fc.bias is None
        if weight is None:
            weight = fc.weight.t()
        else:
            weight = fc(weight)
    return weight


def init_model(model, hidden_sizes, initialization, init_scale=0.01):
    depth = len(hidden_sizes) - 1

    if initialization == 'orthogonal':
        scale = (init_scale * np.sqrt(hidden_sizes[0])) ** (1. / depth)
        matrices = []
        for param in model.parameters():
            nn.init.orthogonal_(param)
            param.data.mul_(scale)
            matrices.append(param.data.cpu().numpy())
        for a, b in zip(matrices, matrices[1:]):
            assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-6)

    elif initialization == 'identity':
        scale = init_scale ** (1. / depth)
        for param in model.parameters():
            nn.init.eye_(param)
            param.data.mul_(scale)

    elif initialization == 'gaussian':
        for param in model.parameters():
            nn.init.normal_(param, std=init_scale)
        e2e = get_e2e(model).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = init_scale * np.sqrt(100)
        print ('[check] e2e fro norm: {}, desired = {}'.format(e2e_fro, desired_fro))

    elif initialization == 'uniform':
        for param in model.parameters():
            scale = np.sqrt(3.) * init_scale ** (1. / depth) * ((param.shape[0] * param.shape[1]) ** (-0.25))
            nn.init.uniform_(param, a=-scale, b=scale)
    else:
        assert 0


class dmf(object):
    def __init__(self, ps):
        self.layers = zip(ps.hidden_sizes, ps.hidden_sizes[1:])
        print(ps.hidden_sizes[0], ps.hidden_sizes[-1])
        self.model = nn.Sequential(*[nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in self.layers]).cuda()


def get_train_loss(predicted_mat, true_mat):
    predicted_vec = predicted_mat.flatten()
    # save indices where original matrix is missing
    idx = ~np.isnan(true_mat.flatten())
    true_vec = torch.from_numpy(true_mat).flatten().cuda()
    loss = nn.MSELoss()
    # assert len(predicted_vec) == len(true_vec)
    return loss(predicted_vec[idx], true_vec[idx])


def get_test_loss(predicted_mat, true_mat, dropped_indices):
    predicted_vec = predicted_mat.flatten()
    idx = dropped_indices
    true_vec = torch.from_numpy(true_mat).flatten().cuda()
    loss = nn.MSELoss()
    return loss(predicted_vec[idx], true_vec[idx])


def select_optimizer(ps, model):
    if ps.optimizer == "SGD":
        return optim.SGD(model.parameters(), ps.lr, momentum=0.9)
    elif ps.optimizer == "Adam":
        return optim.Adam(model.parameters(), ps.lr)
    elif ps.optimizer == "Adagrad":
        return optim.Adagrad(model.parameters(), ps.lr)

def apply_threshold(predicted_mat, original_mat, k):
    # Convert to numpy if tensor
    if torch.is_tensor(predicted_mat):
        pred_np = predicted_mat.detach().cpu().numpy()
    else:
        pred_np = predicted_mat

    # Create mask for non-nan values in original
    valid_mask = ~np.isnan(original_mat)
    # Calculate deviations
    deviations = pred_np - original_mat
    # Create masks for positive and negative deviations exceeding k
    pos_exceed = (deviations > k) & valid_mask
    neg_exceed = (deviations < -k) & valid_mask
    # Create thresholded matrix
    thresholded = pred_np.copy()
    thresholded[pos_exceed] = original_mat[pos_exceed] + k
    thresholded[neg_exceed] = original_mat[neg_exceed] - k
    # Convert back to tensor if input was tensor
    if torch.is_tensor(predicted_mat):
        return torch.from_numpy(thresholded).to(predicted_mat.device)
    return thresholded

def main_function(params, cs=None, logfilename=None, outdir=None):
    ps = param_saver(params, cs)
    dmf_model = dmf(ps)
    init_model(model=dmf_model.model.double(),
               hidden_sizes=ps.hidden_sizes,
               initialization=ps.initialization,
               init_scale=params['init_scale'])

    train_loss_ = []
    test_loss_ = []
    optimizer = select_optimizer(ps, dmf_model.model)
    loss = None
    identity = np.identity(ps.hidden_sizes[0])
    id_input = torch.from_numpy(np.identity(ps.hidden_sizes[0])).double().cuda()

    for T in range(ps.n_iter):
        # initialize a pseudo input
        prediction = dmf_model.model(id_input)
        if params['threshold_type'] == 'in-training' and T % params['threshold_check_freq'] == 0:
            prediction = apply_threshold(prediction, ps.file, params['threshold_k'])
        pred_npy = copy.deepcopy(prediction.cpu().detach().numpy())
        if T % 1000 == 0:
            np.save(os.path.join(outdir, 'pred_npy_{}.npy'.format(T)), pred_npy)
        # mse loss
        train_loss = get_train_loss(prediction, ps.file)
        # total_loss = train_loss
        train_loss_.append(float(train_loss.cpu().detach().numpy()))
        optimizer.zero_grad()
        train_loss.backward()
        with torch.no_grad():
            test_loss = get_test_loss(prediction, ps.A, ps.dropped_idx)
            test_loss_val = float(test_loss.cpu().detach().numpy())
            test_loss_.append(test_loss_val)
        optimizer.step()
        log(logfilename, "{}\t{:.5}\t{:.5}".format(T, train_loss, test_loss))
        if T % 100 == 0:
            print('iteration: {}, train_loss: {}, test_loss: {}'.format(T, train_loss, test_loss))
    
    if params['threshold_type'] == 'post-training':
        final_prediction = apply_threshold(prediction, ps.file, params['threshold_k'])
        final_test_loss = get_test_loss(final_prediction, ps.A, ps.dropped_idx)
        print('Final test loss after thresholding: {}'.format(final_test_loss))
        final_pred_npy = copy.deepcopy(final_prediction.cpu().detach().numpy())
        np.save(os.path.join(outdir, 'final_pred_npy.npy'), final_pred_npy)

    eigen_vector = np.linalg.svd(prediction.cpu().detach().numpy())[1]
    return train_loss_, test_loss_, eigen_vector


def main():
    params = {'hidden_sizes': args.hidden_sizes,
              'n_iter': args.iterations,
              'lr': args.lr,
              'optimizer': args.optimizer,
              'file_dir': args.file_dir,
              'init_scale': args.init_scale,
              'initialization': args.initialization,
              'load_drop_index': True,
              'simulation': False,
              'save_drop_index': True}

    epoch = args.epoch
    outdir = 'synthetic_real_data_v4_{}'.format(args.file_name)

    print('output directory: ', outdir)
    cs = create_simulation(params)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logfilename = os.path.join(outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttrain loss\ttestloss")
    _, _, _ = main_function(params, cs, logfilename, outdir)


if __name__ == "__main__":
    main()
