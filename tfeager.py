#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.17 0017 上午 09:38
@Author  : zhuqingjie 
@User    : zhu
@FileName: tfeager.py
@Software: PyCharm
'''

from threading import Thread
from time import sleep


def async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


def B():
    print("b function")


@async
def A():
    sleep(2)
    print("a function")


def rr():
    A()
    B()
    return 100


if __name__ == '__main__':
    y = rr()
    print(y)
