#!/bin/bash

#SBATCH --job-name=Dysphagia
#SBATCH --mail-type=END
#SBATCH --mail-user=xxxxx@umcg.nl
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.log


# Install:
# # login node
# module purge
# module load CUDA/11.7.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0 
# module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 
# module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0 
# module load magma/2.6.2-foss-2022a-CUDA-11.7.0 
# module load Python/3.8.16-GCCcore-11.2.0
# # module load Python/3.9.5-GCCcore-10.3.0
# python3 -m venv /home4/$USER/.envs/dysphagia_38
# #  interactive node
# source /home4/$USER/.envs/dysphagia_38/bin/activate
# pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# pip3 install pyparsing six python-dateutil
# pip3 install torchinfo tqdm monai pytz SimpleITK pydicom scikit-image matplotlib
# pip3 install torch_optimizer
# pip3 install scikit-learn
# pip3 install timm
# pip3 install wandb
# pip3 install -U numpy
# pip3 install scipy==1.9
# pip3 install pandas
# pip3 install opencv-python
# pip3 install optuna kaleido openpyxl

# Run
module purge
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0 
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0 
module load magma/2.6.2-foss-2022a-CUDA-11.7.0 
module load Python/3.8.16-GCCcore-11.2.0
source /home4/$USER/.envs/dysphagia_38/bin/activate
# srun --gpus-per-node=1 --time=01:00:00 --pty /bin/bash
# srun --gpus-per-node=1  --nodes=1  --cpus-per-task=12  --mem=50G  --time=00:30:00 --pty /bin/bash

# Train
python3 main.py
