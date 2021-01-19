import tensorflow as tf
import os, time, cv2
import config
import numpy as np
from data_init import save2img, merge_smallimgs, read_nd2
from model import UNET_os
from pathlib import Path

# use_GPU = True
#
# if use_GPU:  # BUT! 在这里设置是无效的，还是得在外面设置
#     os.system('module load cuda/10.0')
#     os.system('module load cudnn/7.4.2')
# else:
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 模型目录
model_path = '/home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/model_release/'
print(model_path)


def get_all_nd2datas_to_imgs(nd2_dir='/home/zhangli_lab/zhuqingjie/dataset/optical_section_img/new210115'):
    temp_npy_path = Path(f'{nd2_dir}/xs_ys.npy')
    if temp_npy_path.exists():
        return np.load(temp_npy_path)
    xs, ys = [], []
    nd2files = [f'{nd2_dir}/{i}-w.nd2'
                for i in range(1, 142)]
    nd2files_lb = [f'{nd2_dir}/{i}.nd2'
                   for i in range(1, 142)]
    for xf, yf in zip(nd2files, nd2files_lb):
        xs += read_nd2(xf)
        ys += read_nd2(yf)
    np.save(temp_npy_path, [xs, ys])
    return xs, ys


def predict_os_nd2_data():
    Release_model_path_os = '/home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/model_release/os/'

    flist = os.listdir(Release_model_path_os)
    for f in flist:
        if "model_" in f:
            model_ind = f.split('.')[0]
            break
    model_path_and_ind = Release_model_path_os + model_ind
    print(model_path_and_ind)

    # 保存文件夹,为了方便后续分析，保存在了tunet项目下的目录。
    saved_dir = '/home/zhangli_lab/zhuqingjie/prj/tunet/res_os_nd2/'

    # load datas
    xs, ys = get_all_nd2datas_to_imgs()
    g = UNET_os(predict_flag=True)
    res_imgs = []
    with tf.Session(graph=g.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path_and_ind)

        # 插入，，预测一张并保存
        img = cv2.imread('/home/zhangli_lab/zhuqingjie/DATA/temp/4x.bmp', 0).astype(np.float) / 255
        img = img[None, :, :, None]
        res = sess.run([g.prd], feed_dict={g.x: img})
        res = res[0][0, :, :, 0]
        r_img = res * 255
        r_img = np.round(r_img).astype(np.uint8)
        cv2.imwrite(f'/home/zhangli_lab/zhuqingjie/DATA/temp/4x_output.bmp', r_img)
        exit()

        for i, (x, y) in enumerate(zip(xs, ys)):
            # 先把x和y保存
            x_img = x * 255
            y_img = y * 255
            x_img = np.round(x_img).astype(np.uint8)
            y_img = np.round(y_img).astype(np.uint8)
            cv2.imwrite(f'{saved_dir}{i}_0x.tif', x_img)
            cv2.imwrite(f'{saved_dir}{i}_2y.tif', y_img)

            # 预测
            x = x[None, :, :, None]
            start_time = time.time()
            res = sess.run([g.prd], feed_dict={g.x: x})
            print(f'{i}/{len(xs)} time_use:{time.time() - start_time}')
            res = res[0][0, :, :, 0]
            # res_imgs.append(res)

            # 保存预测结果
            r_img = res * 255
            r_img = np.round(r_img).astype(np.uint8)
            cv2.imwrite(f'{saved_dir}{i}_1deeps.tif', r_img)


if __name__ == '__main__':
    predict_os_nd2_data()
    # get_all_nd2datas_to_imgs()

    print('ok')
