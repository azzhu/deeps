import cv2, random
import numpy as np
import config
import os
from nd2file import ND2MultiDim

# param
hw = config.size
step = config.step
data_path = config.data_path


def read_nd2(p):
    '''
    读一张nd2格式的图像（1024*1024），归一化，返回被分割的4张小图像（512*512）
    :param p:
    :return:
    '''
    nd2 = ND2MultiDim(p)
    img = nd2.image_singlechannel()
    img = img.astype(np.float) / img.max()
    return [
        img[:512, :512],
        img[:512, 512:],
        img[512:, :512],
        img[512:, 512:]
    ]


# 0-1的矩阵保存为0-255的图像
def save2img(arr, path):
    arr = np.reshape(arr, [hw, hw])
    arr = arr * 255
    arr = arr.astype(np.uint8)
    cv2.imwrite(path, arr)


# 不管图像深度是多少 都按8位来算
def get_data():
    datafilelist = sorted(os.listdir(data_path + 'data'))
    labelfilelist = sorted(os.listdir(data_path + 'label'))
    [print(x) for x in zip(datafilelist, labelfilelist)]
    datafilelist = [data_path + 'data/' + x for x in datafilelist]
    labelfilelist = [data_path + 'label/' + x for x in labelfilelist]

    tra_data_list = list(zip(datafilelist, labelfilelist))

    # ind = 1
    data_tra = []
    label_tra = []
    for _ in tra_data_list:
        da, lb = _[0], _[1]
        src = cv2.imread(da, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255.0
        dst = cv2.imread(lb, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255.0
        # 把标签设成输入图像和输出图像的差
        # dst = src - dst
        h, w = src.shape[:2]
        for i in range(0, h - hw, step):
            for j in range(0, w - hw, step):
                data_tra.append(src[i:i + hw, j:j + hw, np.newaxis])
                label_tra.append(dst[i:i + hw, j:j + hw, np.newaxis])
    # 打乱
    ind = list(range(len(data_tra)))
    random.shuffle(ind)
    data_tra = [data_tra[i] for i in ind]
    label_tra = [label_tra[i] for i in ind]

    bs = config.batch_size
    data_tra = [data_tra[i:i + bs] for i in range(0, len(data_tra), bs)]
    label_tra = [label_tra[i:i + bs] for i in range(0, len(label_tra), bs)]
    return np.array(data_tra[:-1]).astype(np.float32), \
           np.array(label_tra[:-1]).astype(np.float32)


# # 不管图像深度是多少 都按8位来算
# # 读取测试图像，只能一张
# def get_test_data():
#     # ind = 1
#     data_tra = []
#     da = '/home/zhuqingjie/prj/tunet_onesample/data/test-x.tif'
#     src = cv2.imread(da, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255.0
#     h, w = src.shape[:2]
#     for i in list(range(0, h - hw, hw)) + [h - hw]:
#         for j in list(range(0, w - hw, hw)) + [w - hw]:
#             data_tra.append(src[i:i + hw, j:j + hw, np.newaxis])
#
#     return np.array(data_tra).astype(np.float32), h, w


# eg:
# input:(None,96,96,1)
# output:(1024,1024)
def merge_smallimgs(imgs, src_h, src_w):
    imgs = imgs[:, :, :, 0]
    n, h, w = imgs.shape[:3]
    n_h = len(list(range(0, src_h - h, h)) + [src_h - h])
    n_w = len(list(range(0, src_w - w, w)) + [src_w - w])
    assert (n_h - 1) * h <= src_h <= n_h * w, 'param src_h error'
    assert (n_w - 1) * w <= src_w <= n_w * w, 'param src_w error'
    assert n_h * n_w == n, 'param error'

    k = 0
    res = np.zeros([src_h, src_w], imgs.dtype)
    for i in list(range(0, src_h - h, h)) + [src_h - h]:
        for j in list(range(0, src_w - w, w)) + [src_w - w]:
            res[i:i + hw, j:j + hw] = imgs[k]
            k += 1
    return res


if __name__ == '__main__':
    img = cv2.imread('/home/zhuqingjie/app/dog.bmp')
    img = np.transpose(img, [2, 0, 1])
    x, h, w = get_test_data()
    y = merge_smallimgs(x, h, w)
    cv2.imwrite('/home/zhuqingjie/yyy.bmp', (y * 255).astype(np.uint8))
    # if '0.' in '10.bmp':
    #     print('ok')
    print()
