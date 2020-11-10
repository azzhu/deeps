#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.05 0005 上午 11:07
@Author  : zhuqingjie
@User    : zhu
@FileName: server_2.py
@Software: PyCharm
'''

import json, os, cv2, sys, time
# import tensorflow as tf
# from threading import Thread
from flask import Flask, request

sys.path.append('../..')
# from train import G
# import config as cf
# import numpy as np
# from SERVER.data_preprocess import process, get_batch
import SERVER.modules.train as train

# # import config as cf
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, cf.gpus)))
#
# # param
# sr_or_os = 'sr'
# model_path = '/home/zhuqingjie/prj/tunet_onesample/model_release'
# saved_dir = f'/GPFS/zhuqingjie/dataset_saved/tunet_onesample/{sr_or_os}'
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


# 传递一些状态信息
class St():
    def __init__(self):
        self.in_training = False


s = St()


# # 异步装饰器
# def async(f):
#     def wrapper(*args, **kwargs):
#         thr = Thread(target=f, args=args, kwargs=kwargs)
#         thr.start()
#
#     return wrapper
#
#
# @async
# def train(userid, epochs=cf.epoch, batch_size=cf.batch_size):
#     '''
#     userid是一个关键的参数，利用它判断训练数据在哪以及生产的模型要保存的地方，
#     所以该函数不需要传入数据路径以及不需要返回模型地址
#     :param userid:
#     :param epochs:
#     :return:
#     '''
#
#     # 读取训练数据
#     data_dir = f'/GPFS/zhuqingjie/dataset_saved/tunet_onesample/{sr_or_os}/users/data_temp'
#     datas_x = np.load(os.path.join(data_dir, f'{userid}_x.npy'))
#     datas_y = np.load(os.path.join(data_dir, f'{userid}_y.npy'))
#
#     # train
#     h, w = datas_y.shape[1:3]
#     g = G(H=h, W=w)
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 0.3
#     with tf.Session(graph=g.graph, config=config) as sess:
#         var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
#         saver = tf.train.Saver(max_to_keep=1, var_list=var_list_G)
#         sess.run(tf.global_variables_initializer())
#
#         for ep in range(epochs):
#             bxs, bys = get_batch(datas_x, datas_y)
#             for batch_xs, batch_ys in zip(bxs, bys):
#                 _, _, gs = sess.run(
#                     [g.train_op_G, g.train_op_D, g.global_step],
#                     feed_dict={g.x: batch_xs, g.y: batch_ys}
#                 )
#             print_(f'epoch:{ep}/{cf.epoch}')
#         saver_path = f'/GPFS/zhuqingjie/dataset_saved/tunet_onesample/{sr_or_os}/users/model_temp/{userid}'
#         os.mkdir(saver_path)
#         saver.save(sess, f'{saver_path}/model')
#     print('train finished.')
#     sess.close()
#     pass


@app.route('/', methods=['POST'])
def handle():
    if s.in_training:
        return json.dumps({
            'status': -5,
            'info': 'No idle GPU, please try again later.',
            'dst_path': 'null',
        })
    else:
        s.in_training = True
        sr_or_os = 'sr'
        saved_dir = f'/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/{sr_or_os}'
        return train.handle(request.form, sr_or_os, saved_dir, s)
    # print('\n')
    # print('-' * 50)
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #
    # # 初始化输出信息
    # status = 0
    # info = 'initialization info'
    #
    # # 这里是个假循环，只是为了利用break特性
    # while True:
    #     # 读取参数
    #     dic_url = request.form
    #     error_param = 'error_param'
    #     src_path = dic_url.get('src_path', error_param)
    #     mode = dic_url.get('mode', error_param)
    #     donotsave = dic_url.get('donotsave', error_param)
    #     userID = dic_url.get('userID', error_param)
    #     print_(f'\n\tsrc_path: {src_path}\n\tmode: {mode}\n\tdonotsave: {donotsave}\n\tuserID: {userID}')
    #     if error_param in [src_path, mode, donotsave, userID]:
    #         status = -1
    #         info = 'params error!'
    #         break
    #
    #     # 检查参数是否正确
    #     xypaths = [p.strip() for p in src_path.split() if p]
    #     if len(xypaths) == 0:
    #         status = -1
    #         info = 'param error: src_path'
    #         break
    #     flagerr_xyp = 0
    #     for xyp in xypaths:
    #         if ',' not in xyp:
    #             flagerr_xyp = 1
    #     if flagerr_xyp == 1:
    #         status = -1
    #         info = 'param error: src_path'
    #         break
    #
    #     try:
    #         # 处理数据及保存数据
    #         process(src_path, userID, sr_or_os, saved_dir, donotsave)
    #     except:
    #         status = -1
    #         info = 'process data error!'
    #         break
    #
    #     # train，训练数据，保存模型
    #     os.system(f"rm -rf /GPFS/zhuqingjie/dataset_saved/tunet_onesample/{sr_or_os}/users/model_temp/{userID}")
    #     train(userID)
    #
    #     info = 'training...'
    #     break
    #
    # # saved model path
    # model_path = f'/GPFS/zhuqingjie/dataset_saved/tunet_onesample/{sr_or_os}/users/model_temp/{userID}'
    # check_path = model_path
    #
    # # return
    # print_(f"\n\treturn:\n\tstatus: {status},\n\tinfo: {info}")
    # print_('done.')
    # return json.dumps({
    #     'status': status,
    #     'info': info,
    #     'check_path': check_path
    # })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9003')
