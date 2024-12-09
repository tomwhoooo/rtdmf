import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import init_logfile, log

parser = argparse.ArgumentParser(description='DBDNMF')
parser.add_argument('--file-dir', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--file-name', default='', type=str, metavar='PATH',
                    help='path to load file')
parser.add_argument('--iterations', default=50000, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('--epoch', default=0, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='learning rate', dest='lr')
parser.add_argument('--hidden-dim', default=100, type=int,
                    help='hidden dimension size')
parser.add_argument('--rank', default=10, type=int,
                    help='rank for matrix factorization')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='weight for matrix factorization component')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPT',
                    help='optimizer')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training')
args = parser.parse_args()

class param_saver(object):
    def __init__(self, params):
        # read in original matrix
        self.A = np.load('./{}/M_{}.npy'.format(params['file_dir'], args.file_name))
        B = self.A.copy()
        print('matrix shape: ', B.shape)
        self.dropped_idx = np.load('./{}/dropped_idx_{}.npy'.format(params['file_dir'], args.file_name))
        print('num of dropped index: ', len(self.dropped_idx))
        B.ravel()[self.dropped_idx] = np.nan
        self.file = B
        
        # Store both dimensions
        self.input_dim = self.A.shape[0]
        self.output_dim = self.A.shape[1]
        
        self.n_iter = params['n_iter']
        self.lr = params['lr']
        self.optimizer = params['optimizer']
        self.hidden_dim = params['hidden_dim']
        self.rank = params['rank']
        self.alpha = params['alpha']

class DBDNMF(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, rank):
        super(DBDNMF, self).__init__()
        # Neural network operates on L (input_dim x rank)
        self.nn = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output should match final matrix dimension
        )
        
        self.L = nn.Parameter(torch.randn(input_dim, rank) * 0.01).cuda()
        self.R = nn.Parameter(torch.randn(rank, output_dim) * 0.01).cuda()
        self.alpha = 0.5
        
    def forward(self, x):
        nn_output = self.nn(self.L)  # f(L): input_dim x output_dim
        
        mf_output = torch.mm(self.L, self.R)  # LR: input_dim x output_dim
        
        return (1 - self.alpha) * nn_output + self.alpha * mf_output

def get_train_loss(predicted_mat, true_mat):
    # predicted_mat is already a tensor with grad
    predicted_vec = predicted_mat.view(-1)
    
    idx = ~np.isnan(true_mat.flatten())
    idx_tensor = torch.from_numpy(idx).cuda()
    
    true_tensor = torch.from_numpy(true_mat.flatten()[idx]).double().cuda()
    
    predicted_values = predicted_vec[idx_tensor]
    
    loss = nn.MSELoss()
    return loss(predicted_values, true_tensor)

def get_test_loss(predicted_mat, true_mat, dropped_indices):
    predicted_vec = predicted_mat.view(-1)
    
    # Convert indices to tensor if needed
    if isinstance(dropped_indices, np.ndarray):
        dropped_indices = torch.from_numpy(dropped_indices).long().cuda()
    
    # Get true values
    true_tensor = torch.from_numpy(true_mat.flatten()[dropped_indices]).double().cuda()
    
    # Get predicted values with gradient tracking
    predicted_values = predicted_vec[dropped_indices]
    
    loss = nn.MSELoss()
    return loss(predicted_values, true_tensor)


def select_optimizer(ps, model):
    if ps.optimizer == "SGD":
        return optim.SGD(model.parameters(), ps.lr, momentum=0.9)
    elif ps.optimizer == "Adam":
        return optim.Adam(model.parameters(), ps.lr)
    elif ps.optimizer == "Adagrad":
        return optim.Adagrad(model.parameters(), ps.lr)

def main_function(params, logfilename=None, outdir=None):
    ps = param_saver(params)
    model = DBDNMF(ps.input_dim, ps.output_dim, ps.hidden_dim, ps.rank).double().cuda()
    model.alpha = ps.alpha
    
    train_loss_ = []
    test_loss_ = []
    optimizer = select_optimizer(ps, model)
    
    identity = torch.eye(ps.input_dim).double().requires_grad_(True).cuda()

    for T in range(ps.n_iter):
        prediction = model(identity)
        
        if T % 1000 == 0:
            # Detach for saving
            pred_npy = prediction.detach().numpy()
            np.save(os.path.join(outdir, 'pred_npy_{}.npy'.format(T)), pred_npy)
        
        train_loss = get_train_loss(prediction, ps.file)
        train_loss_.append(float(train_loss.detach().numpy()))
        
        optimizer.zero_grad()
        train_loss.backward()
        
        with torch.no_grad():
            test_loss = get_test_loss(prediction, ps.A, ps.dropped_idx)
            test_loss_val = float(test_loss.detach().numpy())
            test_loss_.append(test_loss_val)
        
        optimizer.step()
        
        log(logfilename, "{}\t{:.5}\t{:.5}".format(T, train_loss, test_loss))
        if T % 100 == 0:
            print('iteration: {}, train_loss: {}, test_loss: {}'.format(
                T, train_loss, test_loss))
    
    # Save final prediction
    with torch.no_grad():
        final_pred = prediction.numpy()
        np.save(os.path.join(outdir, 'final_pred.npy'), final_pred)
    
    return train_loss_, test_loss_

def main():
    params = {
        'n_iter': args.iterations,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'file_dir': args.file_dir,
        'hidden_dim': args.hidden_dim,
        'rank': args.rank,
        'alpha': args.alpha
    }

    torch.manual_seed(args.seed)

    outdir = './{}/results_dbdnmf2_{}_alpha{}_seed{}'.format(params['file_dir'], args.file_name, params['alpha'], args.seed)

    print('output directory: ', outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logfilename = os.path.join(outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttrain loss\ttestloss")

    train_losses, test_losses = main_function(params, logfilename, outdir)

if __name__ == "__main__":
    main()