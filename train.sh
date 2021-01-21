#!/bin/sh

module load cuda/10.0
module load cudnn/7.4.2


srun -J train_zqj -p q_ai -c 18 --gres=gpu:1 /home/zhangli_lab/zhuqingjie/DATA/env/py3/bin/python train.py