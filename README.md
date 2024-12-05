# Residual Thresholded Deep Matrix Factorization (RT-DMF) Implementation

This repository contains an implementation of RT-DMF and CSV data support.

## Setup

Create and activate virtual environment, then install requirements.

```bash
python -m venv dmf_env

# For Windows:
dmf_env\Scripts\activate
# For Unix or MacOS:
source dmf_env/bin/activate

pip install -r requirements.txt
```

## Basic Usage

### 1. Parameters Explanation

The main script (`run_rtdmf.py`) accepts the following parameters:

#### Network Architecture Parameters

- `--hidden-sizes` (--hs): List of integers defining network layer sizes
  - Example: `--hidden-sizes 500 500 500` creates a network with three hidden layers
  - The input and output dimensions are automatically added based on data size

#### Training Parameters

- `--iterations`: Number of training iterations (default: 100000)
- `--lr`: Learning rate (default: 0.00001)
- `--optimizer`: Choice of optimizer (default: 'Adam')
  - Options: 'SGD', 'Adam', 'Adagrad'
- `--momentum`: Momentum for SGD optimizer (default: 0.9)

#### Initialization Parameters

- `--init-scale`: Initial scale for network weights (default: 0.001)
- `--initialization`: Weight initialization method (default: 'gaussian')
  - Options: 'gaussian', 'identity', 'orthogonal', 'uniform'

#### Thresholding Parameters

- `--threshold-type`: Type of thresholding ('none', 'in-training', 'post-training'). Please refer to the paper for more details about thresholding. (default: 'none')
- `--threshold-k`: Maximum allowed deviation from original values (default: 0.1)
- `--threshold-check-freq`: Frequency of threshold checking for in-training mode (default: 10000)

#### Other Parameters

- `--file-name`: Name of the input file
- `--file-dir`: Directory containing input files
- `--gpu`: GPU ID to use (default: 0)
- `--seed`: Random seed for reproducibility

### 2. Running the Code

Basic example:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python run_rtdmf.py \
    --file-name example_data \
    --hidden-sizes 500 500 500 \
    --lr 0.0001 \
    --threshold-type in-training \
    --threshold-k 0.5
```

You may take a look at our sample shell scripts for more examples.

## Using with CSV Data

If you have a CSV file, use the helper script (`csv_helper.py`) to prepare your data - converting it to the numpy format used by RT-DMF.

### 1. Prepare Your CSV Data

Your CSV should:

- Contain numeric data
- Have consistent number of columns
- Use standard missing value indicators (NA, NaN, empty cells, etc.)

### 2. Convert CSV to DMF Format

Basic usage:

```bash
python csv_helper.py \
    --csv-path your_data.csv
```

For people who prepare their data in non-standard ways (such as certain industrial data formats), you can specify custom NA values. If you want to use custom NA values:

```bash
python prepare_csv_dmf.py \
    --csv-path your_data.csv \
    --custom-na-values "missing" "999" \
    --output-dir my_data_dir
```

### 3. Run DMF on Prepared Data

After preparing your CSV, run DMF using the command provided by the helper script.

## Output Files and Monitoring

### Directory Structure

Your directory will look like something like this (here we assume the output directory is `synthetic_real_data_v4_[filename]`):

```
synthetic_real_data_v4_[filename]/
├── log.txt                  # Training progress log
├── pred_npy_[iteration].npy # Predictions saved every 1000 iterations
└── final_pred.npy          # Final predictions after training
```

### Monitoring Progress

- Check `log.txt` for training and test losses
- Monitor predictions in `pred_npy_[iteration].npy` files
- Final results in `final_pred.npy`

## Reproducing Baselines

The `Baselines` directory contains the code for the baselines used in the paper. You can run the baselines following the instructions in the corresponding shell scripts. You can reproduce the graphs in the paper using the `plot_xxx.py` scripts (where `xxx` is the name of the baseline).

## Common Issues and Solutions

1. **Memory Issues**:
   - Reduce network size
   - Use smaller learning rate
   - Process data in chunks

2. **Convergence Issues**:
   - Adjust learning rate
   - Try different initializations
   - Increase number of iterations

3. **NaN Values**:
   - Check input data normalization
   - Reduce learning rate
   - Monitor gradient values

For more detailed information or issues, please refer to the paper or create an issue in the repository.
