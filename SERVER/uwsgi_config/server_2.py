#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.05 0005 上午 11:07
@Author  : zhuqingjie
@User    : zhu
@FileName: server_2.py
@Software: PyCharm
'''

import sys
from flask import Flask, request

sys.path.append('../..')
import SERVER.modules.inference as inference
import SERVER.someconfig as cfg

app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle():
    model_path = cfg.Release_model_path_sr
    saved_dir = cfg.Saved_dir_sr
    return inference.handle(request.form, model_path, saved_dir, 'sr')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9002')
