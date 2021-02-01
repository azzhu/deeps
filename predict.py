import tensorflow as tf
import os, time, cv2
import config
import numpy as np
from data_init import merge_smallimgs, read_nd2, get_all_nd2datas_to_imgs, Datas_nd2
from model import UNET_sr as G
from pathlib import Path

# 模型目录
# model_path = '/home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/model_release/'
# print(model_path)

'''
os justunet模型路径：
'/home/zhangli_lab/zhuqingjie/prj/tunet_onesample/logdir_nd2_justunet/model_006300'
'''

def predict_os_nd2_data():
    model_path_dir = '/home/zhangli_lab/zhuqingjie/prj/tunet_onesample/logdir_nd2_justunet/'
    # model_path_dir = '/home/zhangli_lab/zhuqingjie/prj/tunet_onesample/logdir_nd2_deeps/'

    flist = list(Path(model_path_dir).rglob('*.index'))
    key_fun = lambda x: int(x.stem.split('_')[1])
    flist = sorted(flist, key=key_fun)
    model_path_and_ind = str(Path(flist[-1].parent, flist[-1].stem))
    # model_path_and_ind = '/home/zhangli_lab/zhuqingjie/prj/tunet_onesample/logdir_nd2_justunet/model_006300'
    print(model_path_and_ind)
    # exit()

    # 保存文件夹,为了方便后续分析，保存在了tunet项目下的目录。
    saved_dir = '/home/zhangli_lab/zhuqingjie/prj/tunet/res_os_nd2/'

    # load datas
    test_datas = Datas_nd2.load_test_datas_512()
    test_datas = np.squeeze(test_datas)
    g = G(predict_flag=True)
    with tf.Session(graph=g.graph) as sess:
        saver = tf.train.Saver()
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path_and_ind)
        # exit()

        # # 插入，，预测一张并保存
        # img = cv2.imread('/home/zhangli_lab/zhuqingjie/DATA/temp/4x.bmp', 0).astype(np.float) / 255
        # img = img[None, :, :, None]
        # res = sess.run([g.prd], feed_dict={g.x: img})
        # res = res[0][0, :, :, 0]
        # r_img = res * 255
        # r_img = np.round(r_img).astype(np.uint8)
        # cv2.imwrite(f'/home/zhangli_lab/zhuqingjie/DATA/temp/4x_output.bmp', r_img)
        # exit()

        for i, (x, y) in enumerate(test_datas):
            # # 先把x和y保存
            # x_img = x * 255
            # y_img = y * 255
            # x_img = np.round(x_img).astype(np.uint8)
            # y_img = np.round(y_img).astype(np.uint8)
            # cv2.imwrite(f'{saved_dir}{i}_0x.tif', x_img)
            # cv2.imwrite(f'{saved_dir}{i}_4y.tif', y_img)

            # 预测
            x = x[None, :, :, None]
            start_time = time.time()
            res = sess.run([g.prd], feed_dict={g.x: x})
            print(f'{i}/{len(test_datas)} time_use:{time.time() - start_time}')
            res = res[0][0, :, :, 0]
            # res_imgs.append(res)

            # 保存预测结果
            r_img = res * 255
            r_img = np.round(r_img).astype(np.uint8)
            cv2.imwrite(f'{saved_dir}{i}_2unet.tif', r_img)
            # cv2.imwrite(f'{saved_dir}{i}_3deeps.tif', r_img)


if __name__ == '__main__':
    predict_os_nd2_data()
    # get_all_nd2datas_to_imgs()

    print('ok')
