#!/bin/bash

#SBATCH -J train_j
#SBATCH -o out.out
#SBATCH -p q_ai
#SBATCH -c 3
#SBATCH --gres=gpu:1


module load cuda/10.0
module load cudnn/7.4.2

/home/zhangli_lab/zhuqingjie/DATA/env/py3/bin/python train.py