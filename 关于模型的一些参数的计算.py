#! python
# @Time    : 21/01/11 下午 03:50
# @Author  : azzhu 
# @FileName: 关于模型的一些参数的计算.py
# @Software: PyCharm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model import UNET_os, UNET_sr
import tensorflow as tf
from pathlib import Path
import random
import cv2
import numpy as np
import time


# import tensorflow.python.framework.ops as ops

def 统计FLOPs和模型参数():
    Net = UNET_sr
    # Net = UNET_os

    g = Net(predict_flag=False, H=256, W=256)

    with g.graph.as_default():
        variable = tf.trainable_variables()
        print()
        flops = tf.profiler.profile(g.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print(f'{Net.__name__} FLOPs: {flops.total_float_ops}')

        # 参数量
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(f'total_parameters: {total_parameters}')


def 统计模型推理时间():
    imgfiles = list(Path(f'/home/zhangli_lab/zhuqingjie/DATA/temp_guoyunfei/C1_500').rglob('*.tif'))

    # Net = UNET_sr
    Net = UNET_os

    # g = Net(predict_flag=True, H=512, W=512)
    g = Net(predict_flag=True, H=1024, W=1024)
    with tf.Session(graph=g.graph) as sess:
        saver = tf.train.Saver()
        # checkpoint_path = r'/home/zhangli_lab/zhuqingjie/prj/tunet_onesample/model_release/sr/model_68900'
        checkpoint_path = r'/home/zhangli_lab/zhuqingjie/prj/tunet_onesample/model_release/os/model_20700'
        saver.restore(sess, checkpoint_path)

        ts = []
        for imgf in imgfiles:
            img = cv2.imread(str(imgf), 0)
            # img = cv2.resize(img, (512, 512))
            img = cv2.resize(img, (1024, 1024))
            img = img.astype(np.float) / 255
            # img = np.ones((256, 256), np.float)
            img = img[None, :, :, None]
            s_t = time.time()
            prd = sess.run([g.prd], feed_dict={g.x: img})
            e_t = time.time()
            ts.append(e_t - s_t)
            print(e_t - s_t)
            # break
        print('____________')
        ts = np.array(ts[1:])  # 除去第一个的异常值
        np.save('/home/zhangli_lab/zhuqingjie/DATA/temp/ts_os.npy', ts)
        print(np.mean(ts))

    # 结果 ****************************
    """
    平均耗时：
    单张图像(1024*1024)推理平均耗时(硬件：单卡NVIDIA Tesla V100 GPU)：
    os: 0.13212160070339043
    sr: 0.15488938291469415
    
    OS model total parameters: 31914998
    SR model total parameters: 31643574
    OS model FLOPs: 96613461
    SR model FLOPs: 81081997

    """


if __name__ == '__main__':
    统计模型推理时间()
