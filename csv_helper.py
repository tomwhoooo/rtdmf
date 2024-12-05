import numpy as np
import pandas as pd
import os

def prepare_csv_for_dmf(csv_path, output_dir='synthetic_real_data_v4', 
                       custom_na_values=None, keep_default_na=True):
    """
    Prepare CSV data for DMF with flexible NA handling
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    output_dir : str
        Directory to save the prepared data
    custom_na_values : list or str, optional
        Additional values to consider as NA
    keep_default_na : bool, default True
        Whether to keep pandas' default NA values
    
    Returns:
    --------
    dict with keys:
        'matrix_shape': shape of the data matrix
        'missing_count': number of missing values
        'output_files': list of created files
        'na_value_counts': dictionary of NA value counts by type
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read CSV file with NA handling
    df = pd.read_csv(csv_path, 
                     na_values=custom_na_values,
                     keep_default_na=keep_default_na)
    
    # Get NA value information before conversion
    na_info = {
        'total_na_count': df.isna().sum().sum(),
        'na_by_column': df.isna().sum().to_dict()
    }
    
    # Convert data to numpy array
    data = df.to_numpy()
    
    # Get indices of missing values
    mask = pd.isna(data)
    dropped_idx = np.where(mask.flatten())[0]
    
    # Convert data to float, missing values are automatically converted to np.nan
    data = data.astype(float)
    
    # Save files
    matrix_file = os.path.join(output_dir, f'M_user_data.npy')
    dropped_idx_file = os.path.join(output_dir, f'dropped_idx_user_data.npy')
    
    np.save(matrix_file, data)
    np.save(dropped_idx_file, dropped_idx)
    
    return {
        'matrix_shape': data.shape,
        'missing_count': len(dropped_idx),
        'output_files': [matrix_file, dropped_idx_file],
        'na_information': na_info
    }


def suggest_dmf_parameters(matrix_shape):
    """
    Suggest DMF parameters based on matrix size.
    
    Parameters:
    -----------
    matrix_shape : tuple
        Shape of the input matrix
    
    Returns:
    --------
    dict with suggested parameters
    """
    max_dim = max(matrix_shape)
    
    # Suggest hidden layer sizes based on matrix dimensions
    if max_dim <= 100:
        hidden_sizes = [50, 50, 50]
    elif max_dim <= 500:
        hidden_sizes = [200, 200, 200]
    else:
        hidden_sizes = [500, 500, 500]
    
    # Suggest learning rate based on matrix size
    lr = 0.0001 if max_dim <= 500 else 0.00001
    
    return {
        'hidden_sizes': hidden_sizes,
        'learning_rate': lr,
        'iterations': 100000,
        'initialization': 'gaussian',
        'optimizer': 'Adam'
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare CSV data for RT-DMF')
    parser.add_argument('--csv-path', required=True, 
                        help='Absolute path to your CSV file')
    parser.add_argument('--output-dir', default='output', 
                        help='Output directory for prepared data')
    parser.add_argument('--custom-na-values', nargs='+', default=None,
                        help='Additional values to be considered as NA in your dataset')
    parser.add_argument('--ignore-default-na', action='store_true',
                        help='Ignore pandas default NA values (Only if your dataset has a very special N/A representation, likely due to some SPSS processing)')
    
    args = parser.parse_args()
    
    # Prepare data
    result = prepare_csv_for_dmf(
        args.csv_path,
        args.output_dir,
        custom_na_values=args.custom_na_values,
        keep_default_na=not args.ignore_default_na
    )
    
    # Suggested parameters
    params = suggest_dmf_parameters(result['matrix_shape'])
    
    # Print detailed NA information
    print("\nData preparation completed!")
    print(f"Matrix shape: {result['matrix_shape']}")
    print(f"\nMissing value information:")
    print(f"Total missing values: {result['na_information']['total_na_count']}")
    print("\nMissing values by column:")
    for col, count in result['na_information']['na_by_column'].items():
        print(f"  {col}: {count}")
    print("\nSuggested DMF parameters:")
    print(f"Hidden layer sizes: {params['hidden_sizes']}")
    print(f"Learning rate: {params['learning_rate']}")
    print(f"Iterations: {params['iterations']}")
    
    # Generate shell command
    hidden_sizes_str = ' '.join(map(str, params['hidden_sizes']))
    shell_cmd = (f"PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python run_dmf.py "
                f"--file-name user_data "
                f"--hidden-sizes {hidden_sizes_str} "
                f"--lr {params['learning_rate']} "
                f"--iterations {params['iterations']} "
                f"--initialization {params['initialization']} "
                f"--optimizer {params['optimizer']}")
    
    print("\nRun this command to start DMF:")
    print(shell_cmd)