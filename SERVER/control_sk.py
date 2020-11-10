#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.26 0026 下午 01:19
@Author  : zhuqingjie 
@User    : zhu
@FileName: control.py
@Software: PyCharm
'''
'''
socket控制逻辑
'''

import json, os, requests, sys, time
import socket
sys.path.append('..')
from color import Colored as C

print_ = lambda x: print(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}")


def handle():
    print('\n')
    print('-' * 50)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    sk = socket.socket()
    ip = '127.0.0.1'
    port = 10005
    sk.bind((ip, port))
    sk.listen(10)
    while True:
        new_cil, addr = sk.accept()
        print(f'new addr:{addr}')
        print(new_cil.recv(1024).decode())
        new_cil.send(b'huixin haha')
        new_cil.send(b'zai huixin hahaha')
        new_cil.close()


if __name__ == '__main__':
    # handle()
    print_ = lambda x: print(f'\033[1;32;0m{x}\033[0m')
    print_('nihaoma beijing?')
    print('\033[1;32;0mnihaoma beijing?\033[0m')
    print(C.blue('nihaoma beijing?'))
