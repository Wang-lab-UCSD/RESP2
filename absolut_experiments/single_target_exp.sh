#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name strs
#SBATCH --output strs
#SBATCH -w gpu-2
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

source ~/.bashrc
source ~/.bash_profile

conda activate resp_absolut

python run_absolut_experiments.py --run_single_target_resp_search
