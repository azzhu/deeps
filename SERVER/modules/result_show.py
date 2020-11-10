#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3/bin/python
'''
@Time    : 11.19 0019 上午 10:22
@Author  : zhuqingjie 
@User    : zhu
@FileName: result_show.py
@Software: PyCharm
'''
import cv2, random, os, json, time, sys
import numpy as np

sys.path.append('..')
from color import Colored as C

print_ = lambda x: print(C.blue(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}"))


def img_overlay_intensity_line(*imgs_, **kwargs):
    '''
    在图像中随机选取一条横线，然后在其上方画出强度值曲线。目前只处理灰度图，传彩色图也按灰度图处理，返回也是灰度图
    :param imgs_: 多张要处理的图像,可以是numpy矩阵形式也可以是路径形式，矩阵返回矩阵，路径返回路径
    :param kwargs:
    :return: 返回多张图像相同坐标下的强度分布曲线图
    '''
    try:
        # print(imgs_)
        # print(kwargs)
        dist = int(kwargs.get('dist', 150))  # 要统计的直线的长度
        high = int(kwargs.get('high', 80))  # 展示的时候曲线最高的高度
        smk = int(kwargs.get('smooth_ker', 7))  # 平滑直线的卷积核的大小
        thichness = int(kwargs.get('thichness', 2))  # 曲线粗细度
        # print(dist, high, smk, thichness)
        # print(type(dist))

        if type(imgs_[0]) is str:
            str_flag = True
        else:
            str_flag = False
        if str_flag:
            imgs = [cv2.imread(i, 0) for i in imgs_]
        else:
            imgs = imgs_

        h, w = imgs[0].shape[:2]
        # 左边点坐标范围
        h_min, h_max = 3 + high, h - 3
        w_min, w_max = 3, w - 3 - dist
        # 随机选择一个点（左边的点）
        lp = (random.randint(h_min, h_max), random.randint(w_min, w_max))
        # 对应的右边的点
        rp = (lp[0], lp[1] + dist)

        res = []
        r = high / 255.
        for img in imgs:
            value_list = img[lp[0], lp[1]:rp[1]]
            k = int(smk / 2)
            vs = np.pad(value_list, (k, k), 'reflect')
            vs_sm = [np.mean(vs[i - k:i + k + 1]) for i in range(k, k + dist)]
            draw_ps = [(lp[0] - int(vs_sm[i] * r), lp[1] + i) for i in range(dist)]
            # for p in draw_ps:  img[p[0], p[1]] = 255
            img = np.stack((img, img, img), axis=-1)
            for i in range(dist - 1):
                cv2.line(img, (draw_ps[i][1], draw_ps[i][0]), (draw_ps[i + 1][1], draw_ps[i + 1][0]), (0, 0, 255),
                         thichness)
            cv2.circle(img, (lp[1], lp[0]), thichness + 1, (0, 0, 255), -1)
            cv2.circle(img, (rp[1], rp[0]), thichness + 1, (0, 0, 255), -1)
            img = img.astype(np.int16)
            img[lp[0], lp[1]:rp[1], 2] += 120
            img[lp[0] - high:lp[0], lp[1], 1] += 120
            img[rp[0] - high:rp[0], rp[1], 1] += 120
            img[lp[0] - high, lp[1]:rp[1], 1] += 120
            img = np.clip(img, 0, 255).astype(np.uint8)
            # cv2.line(img, (lp[1], lp[0]), (rp[1], rp[0]), 255, 1)
            res.append(img)
        if str_flag:
            res_ = []
            for img, path in zip(res, imgs_):
                filename, ext = os.path.splitext(path)
                path_new = filename + '_IOIL' + ext
                cv2.imwrite(path_new, img)
                res_.append(path_new)
            return res_
        else:
            return res
    except:
        print(f'{sys._getframe().f_code.co_name}:excepted!')
        return imgs_


def two_imgs_diffs(*imgs_):
    try:
        if type(imgs_[0]) is str:
            str_flag = True
        else:
            str_flag = False

        if str_flag:
            imgs = [cv2.imread(i, 0) for i in imgs_]
        else:
            imgs = imgs_

        imgs = [i.astype(np.int16) for i in imgs]
        diffimg = np.abs(imgs[0] - imgs[1])
        hot_diffimg = cv2.applyColorMap(diffimg.astype(np.uint8), 8)
        if str_flag:
            n0 = os.path.splitext(os.path.basename(imgs_[0]))[0]
            n1 = os.path.splitext(os.path.basename(imgs_[0]))[0]
            dr = os.path.dirname(imgs_[0])
            path_new = os.path.join(dr, f'{n0}_{n1}.bmp')
            cv2.imwrite(path_new, hot_diffimg)
            return path_new
        else:
            return hot_diffimg

    except:
        print(f'{sys._getframe().f_code.co_name}:excepted!')
        return imgs_


def handle(dic_url):
    print('\n')
    print('-' * 50)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)

    # 初始化输出信息
    status = 0
    info = 'initialization info'
    dst_path = 'none'

    # 这里是个假循环，只是为了利用break特性
    while True:
        # 读取参数
        error_param = 'error_param'
        mode = dic_url.get('mode', error_param)
        src_path = dic_url.get('src_path', error_param)
        print_(f'\n\tmode: {mode}\n\tsrc_path: {src_path}')
        if error_param in [mode, src_path]:
            status = -1
            info = 'params error!'
            break

        if mode in ['10', 10]:
            imgs = src_path.split()
            # print(imgs, dic_url)
            res = img_overlay_intensity_line(*imgs, **dic_url.to_dict())
            dst_path = ' '.join(res)
            info = 'success'
            break
        elif mode in ['11', 11]:
            imgs = src_path.split()
            # print(imgs)
            dst_path = two_imgs_diffs(*imgs)
            info = 'success'
            break
        else:
            status = -1
            info = 'params error!'
            break

    # return
    print_(f"\n\treturn:\n\tstatus: {status},\n\tinfo: {info},\n\tdst_path: {dst_path}")
    print_('done.')
    return json.dumps({
        'status': status,
        'info': info,
        'dst_path': dst_path
    })


if __name__ == '__main__':
    # f1 = '/home/zhangli_lab/zhuqingjie/temp/10x0081.tif /home/zhangli_lab/zhuqingjie/temp/10y0081.tif'
    # f2 = '/home/zhangli_lab/zhuqingjie/temp/10y0081.tif'
    # f = [f1, f2]
    d = {'dist': 300, 'high': 200, 'thickness': 1, 'smk': 7}
    dic = {'mode': 10,
           'src_path': '/home/zhangli_lab/zhuqingjie/temp/10x0081.tif /home/zhangli_lab/zhuqingjie/temp/10y0081.tif'}
    dic.update(d)
    x = handle(dic)
    # print(img_overlay_intensity_line(*f, **d))
    print()
