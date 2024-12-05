import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Neural Network + MF params')
parser.add_argument('--block-size', default=50, type=int,
                    help='block size')
parser.add_argument('--p', default=0.05, type=float,
                    help='p')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='weight for matrix factorization component')
parser.add_argument('--hidden-dim', default=100, type=int,
                    help='hidden dimension size')
parser.add_argument('--rank', default=10, type=int,
                    help='rank for matrix factorization')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of epochs')
args = parser.parse_args()

class DBDNMF(nn.Module):
    def __init__(self, input_dim, hidden_dim, rank):
        super(DBDNMF, self).__init__()
        # Neural Network part
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Matrix Factorization part
        self.L = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.R = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        
    def forward(self, x):
        nn_output = self.nn(x)
        mf_output = torch.mm(self.L, self.R)
        return nn_output + args.alpha * mf_output

def main():
    # Load data
    Block_size = args.block_size
    p_ = args.p
    N_noise = 0.05
    
    M = np.load(f'./synthetic_real_data_v4/M_B{Block_size}_noise{N_noise}_p{p_}.npy')
    not_nan_idx = np.load(f'./synthetic_real_data_v4/not_nan_idx_B{Block_size}_noise{N_noise}_p{p_}.npy')
    dropped_idx = np.load(f'./synthetic_real_data_v4/dropped_idx_B{Block_size}_noise{N_noise}_p{p_}.npy')
    
    Mask = np.zeros_like(M)
    non_dropped_idx = np.setdiff1d(not_nan_idx, dropped_idx)
    Mask.ravel()[non_dropped_idx] = 1
    
    M_tensor = torch.FloatTensor(M)
    Mask_tensor = torch.FloatTensor(Mask)
    
    input_dim = M.shape[0]
    model = DBDNMF(input_dim, args.hidden_dim, args.rank).cuda()
    M_tensor = M_tensor.cuda()
    Mask_tensor = Mask_tensor.cuda()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    losses = []
    identity = torch.eye(input_dim).cuda()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Forward pass
        output = model(identity)
        
        # Calculate loss only on non-missing values
        loss = criterion(output * Mask_tensor, M_tensor * Mask_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')
        losses.append(loss.item())
    
    output_dir = 'dbdnmf_results_v4'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    final_output = model(identity).detach().cpu().numpy()
    np.save(
        f'./{output_dir}/pred_B{Block_size}_noise{N_noise}_p{p_}_alpha{args.alpha}_rank{args.rank}.npy',
        final_output
    )
    np.save(
        f'./{output_dir}/losses_B{Block_size}_noise{N_noise}_p{p_}_alpha{args.alpha}_rank{args.rank}.npy',
        np.array(losses)
    )
    
if __name__ == "__main__":
    main()