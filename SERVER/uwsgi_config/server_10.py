#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 11.22 0005 下午 03:55
@Author  : zhuqingjie
@User    : zhu
@FileName: server_10.py
@Software: PyCharm
'''

import os, sys, time
from flask import Flask, request

sys.path.append('../..')
import SERVER.modules.result_show as show

os.environ['CUDA_VISIBLE_DEVICES'] = ''

app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle():
    return show.handle(request.form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9010')
