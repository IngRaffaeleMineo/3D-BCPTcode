call C:/Anaconda3/Scripts/activate.bat MY_ENVIROMENT

python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env train.py --split_path=data\\dataset.json -- num_fold=0
python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env train.py --split_path=data\\dataset.json -- num_fold=1
python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env train.py --split_path=data\\dataset.json -- num_fold=2
python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env train.py --split_path=data\\dataset.json -- num_fold=3
python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env train.py --split_path=data\\dataset.json -- num_fold=4

pause