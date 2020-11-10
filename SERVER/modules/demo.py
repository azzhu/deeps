#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 08.05 0005 下午 01:45
@Author  : zhuqingjie 
@User    : zhu
@FileName: demo.py
@Software: PyCharm
'''

import json, os, cv2, sys, time
import tensorflow as tf

sys.path.append('..')
from color import Colored as C

print_ = lambda x: print(C.blue(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}"))


def handle(dic_url, return_x, return_predict, return_y):
    print('\n')
    print('-' * 50)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)

    # 初始化输出信息
    status = 0
    info = 'initialization info'

    # 这里是个假循环，只是为了利用break特性
    while True:
        # 读取参数
        error_param = 'error_param'
        mode = dic_url.get('mode', error_param)
        print_(f'\n\tmode: {mode}')
        if error_param in [mode]:
            status = -1
            info = 'params error!'
            break

        time.sleep(0.5)
        info = f'{return_x} {return_predict} {return_y}'
        break

    # return
    print_(f"\n\treturn:\n\tstatus: {status},\n\tinfo: {info}")
    print_('done.')
    return json.dumps({
        'status': status,
        'info': info,
    })
