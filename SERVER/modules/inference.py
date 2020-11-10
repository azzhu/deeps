#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 08.05 0005 下午 01:45
@Author  : zhuqingjie 
@User    : zhu
@FileName: inference.py
@Software: PyCharm
'''

import json, os, cv2, sys, time
import tensorflow as tf
from flask import Flask, request
from pathlib import Path

sys.path.append('..')
from color import Colored as C
# from model import UNET as G
import numpy as np

# import config as cf
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print_ = lambda x: print(C.blue(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}"))


def handle(dic_url, model_path, saved_dir, sr_or_os):
    if sr_or_os == 'sr':
        from model import UNET_sr as G
    elif sr_or_os == 'os':
        from model import UNET_os as G
    else:
        print('inference.py: line 25, [sr_or_os] must be "sr" or "os".')
        exit()

    print('\n')
    print('-' * 50)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 初始化输出信息
    status = 0
    info = 'initialization info'
    dst_path = 'null'

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
            status = -1
            info = 'params error!'
            break

        # 检测路径或参数是否合法
        if not os.path.exists(src_path):
            info = 'src_path is not exists!'
            status = -1
            break

        # 读取图像
        # img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(src_path)
        if img.shape[0] <= 70 or img.shape[1] <= 70:
            info = 'the shape of image must greater than 80!'
            status = -1
            break
        if img.shape[0] > 2000 or img.shape[1] > 2000:
            info = 'the shape of image must less than 2000!'
            status = -1
            break

        # 获取要保存的路径
        filename = os.path.basename(src_path)
        path_flag = ''
        if donotsave == '1':
            path_flag = '_undelegated'
        saved_path_x = os.path.join(saved_dir, f'x{path_flag}', filename)
        saved_path_y = os.path.join(saved_dir, f'y{path_flag}', filename)
        if os.path.exists(saved_path_x):
            fn1 = filename.split('.')[0]
            fn2 = filename.split('.')[1]
            ind = 0
            while True:
                if not os.path.exists(os.path.join(saved_dir, f'x{path_flag}', f"{fn1}_{ind}.{fn2}")):
                    saved_path_x = os.path.join(saved_dir, f'x{path_flag}', f"{fn1}_{ind}.{fn2}")
                    saved_path_y = os.path.join(saved_dir, f'y{path_flag}', f"{fn1}_{ind}.{fn2}")
                    break
                ind += 1
        # saved_path_y_tmp = os.path.join(src_path[:src_path.rfind('/')],
        #                                 f'{filename.split(".")[0]}_dst.{filename.split(".")[1]}')
        sp = Path(src_path)
        saved_path_y_tmp = str(Path(sp.parent, f'{sp.stem}_dst{sp.suffix}'))
        print_(f'\n\tsaved_path_x: {saved_path_x}\n\tsaved_path_y: {saved_path_y}')

        # 数据预处理
        H, W = img.shape[:2]
        img = img.astype(np.float32)
        img = img / img.max()
        if len(img.shape) == 3:
            img = np.transpose(img, [2, 0, 1])
            img = img[:, :, :, np.newaxis]
        else:
            img = img[np.newaxis, :, :, np.newaxis]

        # try:
        # 加载图，并推理
        flist = os.listdir(model_path)
        for f in flist:
            if ".meta" in f:
                model_ind = f.split('.')[0]
                break
        checkpoint_path = os.path.join(model_path, model_ind)
        g = G(predict_flag=True, H=H, W=W)
        with tf.Session(graph=g.graph) as sess:
            # var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
            # saver = tf.train.Saver(var_list=var_list_G)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

            prd = sess.run([g.prd], feed_dict={g.x: img})

            # print_('*********************** test time_uesd.')
            # for zqj in range(10):
            #     start_t = time.time()
            #     prd = sess.run([g.prd], feed_dict={g.x: img})
            #     print_(time.time() - start_t)

            dst = prd[0][:, :, :, 0]
            dst = np.transpose(dst, [1, 2, 0])
            dst = dst * 255
            dst = dst.astype(np.uint8)

            # save
            cv2.imwrite(saved_path_y_tmp, dst)
            # os.system(f'cp -p {saved_path_y_tmp} {saved_path_y}')
            os.system(f'cp -p {src_path} {saved_path_x}')
            sess.close()
        dst_path = saved_path_y_tmp
        print_(f'dst_path: {dst_path}')
        info = 'success'
        break
        # except:
        #     status = 1
        #     info = 'inference error!'
        #     break

    # return
    print_(f"\n\treturn:\n\tstatus: {status},\n\tinfo: {info},\n\tdst_path: {dst_path}")
    print_('done.')
    return json.dumps({
        'status': status,
        'info': info,
        'dst_path': dst_path,
    })


if __name__ == '__main__':
    from model import UNET_sr as G

    g = G(predict_flag=True, H=300, W=300)
    with tf.Session(graph=g.graph) as sess:
        var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        saver = tf.train.Saver(var_list=var_list_G)
        saver.restore(sess, '/home/zhangli_lab/zhuqingjie/DATA/prj/tunet/logdir2/model')
        print()
