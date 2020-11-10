import tensorflow as tf
import os, time, cv2
import config
import numpy as np
from data_init import get_test_data, save2img, merge_smallimgs
# from model import UNET as G
from train import G

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 模型目录
model_path = '/home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/model_release/'
print(model_path)


def predict():
    flist = os.listdir(model_path)
    for f in flist:
        if "model_" in f:
            model_ind = f.split('.')[0]
            break
    xs, src_h, src_w = get_test_data()

    g = G(predict_flag=True)
    with tf.Session(graph=g.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path + model_ind)
        start_time = time.time()
        res = sess.run([g.prd], feed_dict={g.x: xs})
        merge_res = merge_smallimgs(res[0], src_h, src_w)
        print('time_use:{}'.format(time.time() - start_time))
        cv2.imwrite(config.data_path + 'prd.bmp', (merge_res * 255.0).astype(np.uint8))


if __name__ == '__main__':
    xx = tf.placeholder(tf.float32, [12, 12, 12, 6])
    xy = tf.placeholder(tf.float32, [12, 13, 13, 6])
    ffff = xx.shape[1] == xy.shape[1]
    zz = tf.cond(tf.constant(True), lambda: xx, lambda: xy)
    # predict()

    print('ok')
