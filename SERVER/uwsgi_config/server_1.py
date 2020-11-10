#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.05 0005 上午 11:07
@Author  : zhuqingjie
@User    : zhu
@FileName: server_2.py
@Software: PyCharm
'''

import json, os, cv2, sys, time
import tensorflow as tf
from flask import Flask, request

sys.path.append('../..')
import SERVER.modules.demo as demo


# from train import G
# import numpy as np

# import config as cf
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# # param
# model_path = '/home/zhuqingjie/prj/tunet_onesample/model_release'
# saved_dir = '/GPFS/zhuqingjie/dataset_saved/tunet_onesample/sr'
#
# flist = os.listdir(model_path)
# for f in flist:
#     if "model_" in f:
#         model_ind = f.split('.')[0]
#         break
# checkpoint_path = os.path.join(model_path, model_ind)
#
# print_ = lambda x: print(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}")

app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle():
    tmpdir = '/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample_temp/pics'
    return_x = os.path.join(tmpdir, 'sr_x.bmp')
    return_predict = os.path.join(tmpdir, 'sr_predict.bmp')
    return_y = os.path.join(tmpdir, 'sr_y.bmp')
    return demo.handle(request.form, return_x, return_predict, return_y)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9001')
