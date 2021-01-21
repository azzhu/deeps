#!/bin/sh

# 加载tensorflow1.14环境
echo 'load cuda and cudnn'
module load cuda/10.0
module load cudnn/7.4.2

# 设置可用GPU
echo 'set GPUs'
export CUDA_VISIBLE_DEVICES="7"

# 运行python脚本
echo 'run'
#/home/zhangli_lab/zhuqingjie/DATA/env/py3/bin/python predict.py
/home/zhangli_lab/zhuqingjie/DATA/env/py3/bin/python train.py