#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.29 0029 下午 01:28
@Author  : zhuqingjie 
@User    : zhu
@FileName: socket_about.py
@Software: PyCharm
'''

import socket


def send(msg):
    mysocket = socket.socket()
    mysocket.connect(('127.0.0.1', 52052))
    mysocket.send(msg)
    mysocket.close()


# err
class Sess():
    def __init__(self, port, ip='127.0.0.1'):
        self.mysocket = socket.socket()
        self.mysocket.connect((ip, port))

    def send(self, msg):
        self.send(msg)

    def __del__(self):
        self.mysocket.close()
