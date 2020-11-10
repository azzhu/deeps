#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 08.05 0005 下午 01:45
@Author  : zhuqingjie 
@User    : zhu
@FileName: train.py
@Software: PyCharm
'''

import json, os, cv2, sys, time
import tensorflow as tf
from threading import Thread

# #################################### 测试GPU能不能用 #############################################
# def get_available_gpus():
#     """
#     code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
#     """
#     from tensorflow.python.client import device_lib as _device_lib
#     local_device_protos = _device_lib.list_local_devices()
#     print('get_available_gpus---------------------')
#     for x in local_device_protos:
#         if x.device_type == 'GPU':
#             print(x.name)
#     # return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
# get_available_gpus()
# print(tf.__version__)
# exit()
# #################################### 测试GPU能不能用 #############################################
'''
注意：tf 1.14 对应cuda版本10.0 对应的cudnn是7.4.2
'''

from flask import Flask, request

sys.path.append('/home/zhangli_lab/zhuqingjie/prj/tunet_onesample')
# from color import Colored as C
# from model import UNET as G
import config as cf
import SERVER.someconfig as somecf
import numpy as np
from SERVER.data_preprocess import process, get_batch

# import config as cf
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, cf.gpus)))

print_ = lambda x: print(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}")
printc = lambda s: print(f"\033[1;35m{s}\033[0m")
printc(tf.__version__)


# 异步装饰器
def async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


@async
def train(userid, sr_or_os, s, epochs=cf.epoch, batch_size=cf.batch_size):
    '''
    userid是一个关键的参数，利用它判断训练数据在哪以及生产的模型要保存的地方，
    所以该函数不需要传入数据路径以及不需要返回模型地址
    :param userid:
    :param epochs:
    :return:
    '''

    # 选择模型拓扑结构
    if sr_or_os == 'sr':
        from model import UNET_sr as G
        model_path = somecf.Release_model_path_sr
    elif sr_or_os == 'os':
        from model import UNET_os as G
        model_path = somecf.Release_model_path_os
    else:
        print('train.py: line 42, [sr_or_os] must be "sr" or "os".')
        exit()

    # 读取训练数据
    data_dir = f'/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/{sr_or_os}/users/data_temp'
    datas_x = np.load(os.path.join(data_dir, f'{userid}_x.npy'))
    datas_y = np.load(os.path.join(data_dir, f'{userid}_y.npy'))
    print_(f'train datas_x.shape:{datas_x.shape}')
    print_(f'train datas_y.shape:{datas_y.shape}')

    # get model path
    flist = os.listdir(model_path)
    for f in flist:
        if ".meta" in f:
            model_ind = f.split('.')[0]
            break
    checkpoint_path = os.path.join(model_path, model_ind)

    # train
    h, w = datas_y.shape[1:3]
    g = G(H=h, W=w)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.33333
    # config.gpu_options.allow_growth = True
    with tf.Session(graph=g.graph, config=config) as sess:
        var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        bn_moving_vars = [g for g in bn_moving_vars if 'generator' in g.name]
        saver = tf.train.Saver(max_to_keep=1, var_list=var_list_G + bn_moving_vars)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        for ep in range(epochs):
            bxs, bys = get_batch(datas_x, datas_y)
            print_(f'get_batch bxs.shape:{bxs.shape}')
            print_(f'get_batch bys.shape:{bys.shape}')
            for batch_xs, batch_ys in zip(bxs, bys):
                _, _, gs = sess.run(
                    [g.train_op_G, g.train_op_D, g.global_step],
                    feed_dict={g.x: batch_xs, g.y: batch_ys}
                )
            print_(f'epoch:{ep}/{cf.epoch}')
        saver_path = f'/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/{sr_or_os}/users/model_temp/{userid}'
        os.mkdir(saver_path)
        saver.save(sess, f'{saver_path}/model')
    print_('train finished.')
    sess.close()
    s.in_training = False
    pass


def handle(dic_url, sr_or_os, saved_dir, s):
    print('\n')
    print('-' * 50)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 初始化输出信息
    status = -1
    info = 'initialization info'

    check_path = 'null'

    # 这里是个假循环，只是为了利用break特性
    while True:
        # 读取参数
        error_param = 'error_param'
        src_path = dic_url.get('src_path', error_param)
        mode = dic_url.get('mode', error_param)
        donotsave = dic_url.get('donotsave', error_param)
        userID = dic_url.get('userID', error_param)
        print_(f'\n\tsrc_path: {src_path}\n\tmode: {mode}\n\tdonotsave: {donotsave}\n\tuserID: {userID}')
        if error_param in [src_path, mode, donotsave, userID]:
            info = 'params error!'
            break

        # 检查参数是否正确
        xypaths = [p.strip() for p in src_path.split() if p]
        if len(xypaths) == 0:
            info = 'param error: src_path'
            break
        flagerr_xyp = 0
        for xyp in xypaths:
            if ',' not in xyp:
                flagerr_xyp = 1
        if flagerr_xyp == 1:
            info = 'param error: src_path'
            break
        # 判断文件是否存在
        existed_flag = 0
        for xyp in xypaths:
            xp, yp = xyp.split(',')[0], xyp.split(',')[1]
            if os.path.exists(xp) and os.path.exists(yp):
                continue
            else:
                existed_flag = 1
                break
        if existed_flag == 1:
            info = 'data error: the files of "src_path" is not existed!'
            break
        # 判断文件shape是否一致,以及，图像是否能读取，图像是否过小或过大
        try:
            shape_flag = 0
            hw_flag = 0
            for xyp in xypaths:
                xp, yp = xyp.split(',')[0], xyp.split(',')[1]
                xp_img = cv2.imread(xp)
                yp_img = cv2.imread(yp)
                h, w = xp_img.shape[:2]
                if xp_img.shape != yp_img.shape:
                    shape_flag = 1
                    break
                if h < 512 or w < 512 or h > 10000 or w > 10000:
                    hw_flag = 1
            if shape_flag == 1:
                info = 'data error: the shape of images is not identical!'
                break
            if hw_flag == 1:
                info = "data error: the size of images is too small or too big! limit:512*512 -> 10000*10000"
                break
        except:
            info = 'data error: read images failed!'
            break

        try:
            # 处理数据及保存数据
            process(src_path, userID, sr_or_os, saved_dir, donotsave)
        except:
            info = 'process data error!'
            break

        # train，训练数据，保存模型
        os.system(
            f"rm -rf /home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/{sr_or_os}/users/model_temp/{userID}")
        train(userID, sr_or_os, s)

        # saved model path
        model_path = f'/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/{sr_or_os}/users/model_temp/{userID}'
        check_path = model_path
        info = 'training...'
        status = 0
        break

    # return
    print_(f"\n\treturn:\n\tstatus: {status},\n\tinfo: {info},\n\tcheck_path: {check_path}")
    print_('done.')
    return json.dumps({
        'status': status,
        'info': info,
        'check_path': check_path
    })


if __name__ == '__main__':
    dic_url = {
        'src_path': '/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample_temp/pics/sr_x.bmp,/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample_temp/pics/sr_predict.bmp',
        'mode': 3,
        'donotsave': 0,
        'userID': 'zhuqingjie_test'
    }


    class St():
        def __init__(self):
            self.in_training = False


    s = St()
    sr_or_os = 'sr'
    saved_dir = f'/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/{sr_or_os}'
    handle(dic_url, sr_or_os, saved_dir, s)
