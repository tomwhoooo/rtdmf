PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_rtdmf.py --file-name B50_noise0.05_p0.0 --hidden-sizes 500 500 500 500 --lr 0.0001 --threshold_type post-training --threshold-k 0.2
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_rtdmf.py --file-name B50_noise0.05_p0.01 --hidden-sizes 500 500 500 500 --lr 0.0001 --threshold_type post-training --threshold-k 0.2
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_rtdmf.py --file-name B50_noise0.05_p0.005 --hidden-sizes 500 500 500 500 --lr 0.0001 --threshold_type post-training --threshold-k 0.2
