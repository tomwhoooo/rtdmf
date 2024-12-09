PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_sbdf_cpu.py --file-dir synthetic_data_realdata --file-name B20_noise0.05_FIMMnoise0.02 \
    --hidden-dim 500 --rank 20 --alpha 0.5 --lr 0.001 --seed 1111

PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_sbdf_cpu.py --file-dir synthetic_data_realdata --file-name B20_noise0.05_FIMMnoise0.02 \
    --hidden-dim 500 --rank 20 --alpha 0.2 --lr 0.001 --seed 1111

PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_sbdf_cpu.py --file-dir synthetic_data_realdata --file-name B20_noise0.05_FIMMnoise0.02 \
    --hidden-dim 500 --rank 20 --alpha 0.4 --lr 0.001 --seed 1111

PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_sbdf_cpu.py --file-dir synthetic_data_realdata --file-name B20_noise0.05_FIMMnoise0.02 \
    --hidden-dim 500 --rank 20 --alpha 0.6 --lr 0.001 --seed 1111

PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python run_sbdf_cpu.py --file-dir synthetic_data_realdata --file-name B20_noise0.05_FIMMnoise0.02 \
    --hidden-dim 500 --rank 20 --alpha 0.8 --lr 0.001 --seed 1111