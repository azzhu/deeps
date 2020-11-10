#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.29 0029 下午 02:29
@Author  : zhuqingjie 
@User    : zhu
@FileName: data_preprocess.py
@Software: PyCharm
'''

'''
面向的场景是：
用户上传一对或少量几对图像（满足一定尺寸的大图像），需要先把这些大图像处理成许多适合网络训练的小图像。
为提高读写效率，小图像采用npy格式保存。
归一化方式：[0-1]

暂不支持彩色图，彩色图也按灰度图处理。但是inference的时候各个通道可以分开处理。
'''

import os, cv2, random, time, sys
import numpy as np
import config as cf

sys.path.append('..')
from color import Colored as C

hw = cf.size
step = cf.step

print_ = lambda x: print(C.blue(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}"))


def process(paths, userid, sr_or_os, saved_dir, donotsave):
    '''
    先是无重叠切（无关step大小），然后再随机选相同数量的
    :param paths: 未被处理的从邵毅那里接收到的数据，字符串，多对路径的话中间用空格隔开。
    eg: '/home/zhuqingjie/dog_x.bmp,/home/zhuqingjie/dog_y.bmp /home/zhuqingjie/cat_x.bmp,/home/zhuqingjie/cat_y.bmp'
    :return:
    '''

    def process_oneimg(imgpath):
        smimgs_x = []
        smimgs_y = []
        if ',' not in imgpath: return smimgs_x, smimgs_y
        imgpath_x = imgpath.split(',')[0]
        imgpath_y = imgpath.split(',')[1]

        # 保存数据
        filename_x = f'data_{os.path.basename(imgpath_x)}'
        filename_y = f'label_{os.path.basename(imgpath_x)}'
        path_flag = ''
        if donotsave == '1':
            path_flag = '_undelegated'
        saved_path_x = os.path.join(saved_dir, f'x{path_flag}', filename_x)
        saved_path_y = os.path.join(saved_dir, f'y{path_flag}', filename_y)
        os.system(f'cp -p {imgpath_x} {saved_path_x}')
        os.system(f'cp -p {imgpath_y} {saved_path_y}')

        try:
            img_x = cv2.imread(imgpath_x, cv2.IMREAD_GRAYSCALE)
            img_x = img_x.astype(np.float32)
            img_x = img_x / img_x.max()
            h_x, w_x = img_x.shape[:2]
            img_y = cv2.imread(imgpath_y, cv2.IMREAD_GRAYSCALE)
            img_y = img_y.astype(np.float32)
            img_y = img_y / img_y.max()
            h_y, w_y = img_y.shape[:2]
            if h_x != h_y or w_x != w_y: return smimgs_x, smimgs_y
            h, w = h_x, w_x
            if h <= hw or w <= hw: return smimgs_x, smimgs_y
            for i in list(range(0, h - hw, hw)) + [h - hw]:
                for j in list(range(0, w - hw, hw)) + [w - hw]:
                    smimgs_x.append(img_x[i:i + hw, j:j + hw, np.newaxis])
                    smimgs_y.append(img_y[i:i + hw, j:j + hw, np.newaxis])
            # 为了增加数据集随机性，再随机选出相同数量的smallimgs
            l = len(smimgs_x)
            for i in range(l):
                p_h = random.randint(0, h - hw)
                p_w = random.randint(0, w - hw)
                smimgs_x.append(img_x[p_h:p_h + hw, p_w:p_w + hw, np.newaxis])
                smimgs_y.append(img_y[p_h:p_h + hw, p_w:p_w + hw, np.newaxis])
            return smimgs_x, smimgs_y
        except:
            return smimgs_x, smimgs_y

    imgpaths = [p.strip() for p in paths.split() if p]
    small_imgs_x = []
    small_imgs_y = []
    for imgpath in imgpaths:
        smimgs_x, smimgs_y = process_oneimg(imgpath)
        small_imgs_x += smimgs_x
        small_imgs_y += smimgs_y
    small_imgs_x = np.array(small_imgs_x)
    small_imgs_y = np.array(small_imgs_y)
    data_dir = f'/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/{sr_or_os}/users/data_temp'
    data_savepath_x = os.path.join(data_dir, f'{userid}_x.npy')
    data_savepath_y = os.path.join(data_dir, f'{userid}_y.npy')
    np.save(data_savepath_x, small_imgs_x)
    np.save(data_savepath_y, small_imgs_y)
    print_(f'shape:{small_imgs_x.shape}')
    print_(f'dtype:{small_imgs_x.dtype}')


def get_batch(xs, ys):
    bs = cf.batch_size
    inds = list(range(len(xs)))
    random.shuffle(inds)
    xs = [xs[i] for i in inds]
    ys = [ys[i] for i in inds]
    if len(xs) % 8 != 0:
        bxs = [xs[i:i + bs] for i in range(0, len(xs), bs)][:-1]
        bys = [ys[i:i + bs] for i in range(0, len(ys), bs)][:-1]
    else:
        bxs = [xs[i:i + bs] for i in range(0, len(xs), bs)]
        bys = [ys[i:i + bs] for i in range(0, len(ys), bs)]
    return np.array(bxs), np.array(bys)


if __name__ == '__main__':
    print(33 % 8)
    exit()
    process('/home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/img_temp/156897141193545200_datafile_1.bmp,'
            '/home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/img_temp/156897141193545200_labelfile_1.bmp',
            'hasud6890', 'os',
            '/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset_saved/tunet_onesample/os/', '0')
    # xs = np.load('/GPFS/zhuqingjie/dataset_saved/tunet_onesample/os/users/data_temp/hasud6890_x.npy')
    # ys = np.load('/GPFS/zhuqingjie/dataset_saved/tunet_onesample/os/users/data_temp/hasud6890_y.npy')
    # bxs, bys = get_batch(xs, ys)
    print()
