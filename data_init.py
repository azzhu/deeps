import cv2, random
import numpy as np
import config
import os, time
from nd2file import ND2MultiDim
from pathlib import Path


# param
# hw = config.size
# step = config.step
# data_path = config.data_path


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
        img[512:, 512:],
    ]


def get_all_nd2datas_to_imgs(
        nd2_dir='/home/zhangli_lab/zhuqingjie/dataset/optical_section_img/new210115',
        file_end_index=241):
    temp_npy_path = Path(f'{nd2_dir}/xs_ys.npy')
    if temp_npy_path.exists():
        return np.load(temp_npy_path)
    xs, ys = [], []
    nd2files = [f'{nd2_dir}/{i}-w.nd2'
                for i in range(1, file_end_index)]
    nd2files_lb = [f'{nd2_dir}/{i}.nd2'
                   for i in range(1, file_end_index)]
    for xf, yf in zip(nd2files, nd2files_lb):
        xs += read_nd2(xf)
        ys += read_nd2(yf)
    np.save(temp_npy_path, [xs, ys])
    return xs, ys


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


class Datas_nd2:
    '''
    【2021.01.20】由于os数据量太少，被reviewer质疑，所以又找郭老师生成了一部分。
    由于没有保存os的justunet模型，所以得重新训，所以deeps也得重新训。
    这个类是用来处理新的数据的。
    '''

    def __init__(self):
        self.data_dir = Path('/home/zhangli_lab/zhuqingjie/dataset/optical_section_img/new210115/')
        # self.__init_nd2_to_npy()
        # self.__datas_split()
        # exit()
        self.load_train_and_test_datas()
        # self.__datas_split_to_256()
        self.train_datas_ids = list(range(len(self.train_datas)))
        self.test_datas_ids = list(range(len(self.test_datas)))

    def __init_nd2_to_npy(self):
        self.xs, self.ys = get_all_nd2datas_to_imgs()

    def __datas_split(self):
        def get_random_ids(train_r=0.75):
            npyf = Path(self.data_dir, f'random_ids_{len(self.xs)}_{train_r}.npy')
            if npyf.exists():
                return np.load(npyf)
            else:
                datalen = len(self.xs)
                trainlen = int(round(datalen * train_r))
                ids = list(range(datalen))
                random.shuffle(ids)
                random.shuffle(ids)
                train_ids = ids[:trainlen]
                test_ids = ids[trainlen:]
                np.save(npyf, [train_ids, test_ids])
                return train_ids, test_ids

        self.train_ids, self.test_ids = get_random_ids()
        xys = np.stack((self.xs, self.ys), 1)
        self.train_datas = xys[self.train_ids][:, :, :, :, None]
        self.test_datas = xys[self.test_ids][:, :, :, :, None]
        np.save(Path(self.data_dir, 'train_datas.npy'), self.train_datas)
        np.save(Path(self.data_dir, 'test_datas.npy'), self.test_datas)

    def __datas_split_to_256(self):
        train = []
        test = []
        for td in self.train_datas:
            st_ps = [0, 80, 160, 240, 256]
            for st_h in st_ps:
                for st_w in st_ps:
                    train.append(td[:, st_h:st_h + 256, st_w:st_w + 256])
        for td in self.test_datas:
            st_ps = [0, 80, 160, 240, 256]
            for st_h in st_ps:
                for st_w in st_ps:
                    test.append(td[:, st_h:st_h + 256, st_w:st_w + 256])
        train = np.array(train)
        test = np.array(test)
        print(train.shape)
        print(test.shape)
        np.save(Path(self.data_dir, 'train_datas_256.npy'), train)
        np.save(Path(self.data_dir, 'test_datas_256.npy'), test)

    def load_train_and_test_datas(self):
        self.train_datas = np.load(Path(self.data_dir, 'train_datas_256.npy'))
        self.test_datas = np.load(Path(self.data_dir, 'test_datas_256.npy'))
        print(f'train: {self.train_datas.shape}')
        print(f'test : {self.test_datas.shape}')

    @staticmethod
    def load_test_datas_512():
        test_datas = np.load(Path('/home/zhangli_lab/zhuqingjie/dataset/optical_section_img/new210115/',
                                  'test_datas.npy'))
        print(f'test : {test_datas.shape}')
        return test_datas

    def get_batch(self, batch_size=12):
        random_ids = random.choices(self.train_datas_ids, k=batch_size)
        das = self.train_datas[random_ids]
        return das[:, 0], das[:, 1]

    def get_batch_test(self, batch_size=12):
        random_ids = random.choices(self.test_datas_ids, k=batch_size)
        das = self.test_datas[random_ids]
        return das[:, 0], das[:, 1]


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
    dn = Datas_nd2()
    st = time.time()
    xs, ys = dn.get_batch()
    print(time.time() - st)
    print(xs.shape, ys.shape)
    exit()
