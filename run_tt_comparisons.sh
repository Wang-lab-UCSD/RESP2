#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name ev_traintest
#SBATCH --output ev_traintest
#SBATCH -w gpu-1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

module load cuda

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

source ../venv_covid2/bin/activate


python run_experiments.py --traintest
