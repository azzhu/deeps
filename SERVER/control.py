#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.26 0026 下午 01:19
@Author  : zhuqingjie 
@User    : zhu
@FileName: control.py
@Software: PyCharm
'''
'''
总的控制逻辑

1，control只向外部暴露一个端口，外部向control发请求，control根据mode来去调用其他server模块
2，同时还解决了外部不能直接访问ai节点的问题。主服务跑在ai节点，control服务跑在登陆节点，这样外部就能访问了
'''

import json, os, requests, sys, time
from flask import Flask, request

# param
ai01_ip = '10.11.1.81'
ai02_ip = '10.11.1.82'
ai03_ip = '10.11.1.83'
ai04_ip = '10.11.1.84'
ai05_ip = '10.11.1.85'
IP = ai05_ip  # 主服务的IP地址

app = Flask(__name__)
print_ = lambda x: print(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}")

printc = lambda s: print(f"\033[1;35m{s}\033[0m")

mode_list = ['1', '2', '21', '22', '3', '4', '5', '51', '6']


def do_request(port, body):
    url = f'http://{IP}:{port}'
    printc(url)
    printc(body)
    response = requests.post(url, data=body)
    printc('do_request ok')
    return response.text


@app.route('/', methods=['POST'])
def handle():
    print('\n')
    print('-' * 50)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 读取参数
    dic_url = request.form
    print_(f'\n\tparams: {dic_url}')
    error_param = 'error_param'
    mode = dic_url.get('mode', error_param)
    if mode == error_param:
        return json.dumps({
            'status': -1,
            'info': 'param error: not find "mode"!',
            'dst_path': 'null',
        })
    elif mode not in mode_list:
        return json.dumps({
            'status': -1,
            'info': 'param error: "mode" must in 1-6!',
            'dst_path': 'null',
        })
    elif mode == '1':
        return do_request(9001, dic_url)
    elif mode == '2':
        return do_request(9002, dic_url)
    elif mode == '21':
        return do_request(9021, dic_url)
    elif mode == '22':
        return do_request(9022, dic_url)
    elif mode == '3':
        return do_request(9003, dic_url)
    elif mode == '4':
        return do_request(9004, dic_url)
    elif mode == '5':
        return do_request(9005, dic_url)
    elif mode == '51':
        return do_request(9051, dic_url)
    elif mode == '6':
        return do_request(9006, dic_url)
    # elif mode in ['10', '11']:
    #     return do_request(9010, dic_url)
    else:
        return json.dumps({
            'status': 2,
            'info': 'error: An impossible error.',
            'dst_path': 'null',
        })


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port='7006')

    body = {
        'mode': '1',
        'donotsave': '0',
        'userID': 'zhuqingj',
        'src_path': '/home/zhangli_lab/zhuqingjie/prj/tunet/res_test/0x.bmp',
    }
    res = do_request(9001, body)
    print(res)
