import inspect
import os

import numpy as np
import tensorflow as tf
import time
import requests, json
# from socket import *
import socket
from time import ctime

# 注意该网络输入值范围是【0-255】
# 使用的时候保证输入从【0-1】归一化到【0-255】

VGG_MEAN = [103.939, 116.779, 123.68]


# input img shape: [None, 224, 224, 3]

def main():
    new_socket = socket.socket()  # 创建 socket 对象
    # ip = "127.0.0.1"  # 获取本地主机名
    ip = "127.0.0.1"  # 获取本地主机名
    port = 52052  # 设置端口
    new_socket.bind((ip, port))  # 绑定端口
    new_socket.listen(5)  # 等待客户端连接并设置最大连接数
    while True:
        new_cil, addr = new_socket.accept()  # 建立客户端连接。
        print('新进来的客户端的地址：', addr)
        # print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        print(new_cil.recv(1024).decode())
        # new_cil.send(b'6')
        new_cil.close()


class Vgg16():
    def __init__(sf):
        vgg16_npy_path = '/home/zhangli_lab/zhuqingjie/DATA/Small_cluster_data/dataset/vgg16.npy'
        data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()

        # load weights
        sf.w = {}
        for key in data_dict:
            sf.w[key] = [
                tf.constant(data_dict[key][0], name=key + '_weight'),
                tf.constant(data_dict[key][1], name=key + '_bias'),
            ]

    def forward(sf, input):
        # input = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 1])
        input_scaled = input * 255.0
        blue = input_scaled - VGG_MEAN[0]
        gree = input_scaled - VGG_MEAN[1]
        red = input_scaled - VGG_MEAN[2]
        bgr = tf.concat([blue, gree, red], axis=-1)

        sf.conv1_1 = sf.conv_layer(bgr, "conv1_1")
        sf.conv1_2 = sf.conv_layer(sf.conv1_1, "conv1_2")
        sf.pool1 = sf.max_pool(sf.conv1_2, 'pool1')

        sf.conv2_1 = sf.conv_layer(sf.pool1, "conv2_1")
        sf.conv2_2 = sf.conv_layer(sf.conv2_1, "conv2_2")
        sf.pool2 = sf.max_pool(sf.conv2_2, 'pool2')

        sf.conv3_1 = sf.conv_layer(sf.pool2, "conv3_1")
        sf.conv3_2 = sf.conv_layer(sf.conv3_1, "conv3_2")
        sf.conv3_3 = sf.conv_layer(sf.conv3_2, "conv3_3")
        return sf.conv3_3
        # sf.pool3 = sf.max_pool(sf.conv3_3, 'pool3')
        #
        # sf.conv4_1 = sf.conv_layer(sf.pool3, "conv4_1")
        # sf.conv4_2 = sf.conv_layer(sf.conv4_1, "conv4_2")
        # sf.conv4_3 = sf.conv_layer(sf.conv4_2, "conv4_3")
        # sf.pool4 = sf.max_pool(sf.conv4_3, 'pool4')
        #
        # sf.conv5_1 = sf.conv_layer(sf.pool4, "conv5_1")
        # sf.conv5_2 = sf.conv_layer(sf.conv5_1, "conv5_2")
        # sf.conv5_3 = sf.conv_layer(sf.conv5_2, "conv5_3")
        # sf.pool5 = sf.max_pool(sf.conv5_3, 'pool5')
        #
        # sf.fc6 = sf.fc_layer(sf.pool5, "fc6")
        # sf.relu6 = tf.nn.relu(sf.fc6)
        # sf.fc7 = sf.fc_layer(sf.relu6, "fc7")
        # sf.relu7 = tf.nn.relu(sf.fc7)
        # sf.fc8 = sf.fc_layer(sf.relu7, "fc8")
        # sf.prob = tf.nn.softmax(sf.fc8, name="prob")

    def avg_pool(sf, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(sf, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(sf, bottom, name):
        with tf.variable_scope(name):
            filt = sf.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = sf.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(sf, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = sf.get_fc_weight(name)
            biases = sf.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_filter(sf, name):
        return sf.w[name][0]

    def get_bias(sf, name):
        return sf.w[name][1]

    def get_fc_weight(sf, name):
        return sf.w[name][0]


if __name__ == '__main__':
    main()
    print('ok.')
