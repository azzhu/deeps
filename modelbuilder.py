#!/home/zhuqingjie/env/py3_tf_low/bin/python
'''
@Time    : 07.08 0008 下午 07:25
@Author  : zhuqingjie 
@User    : zhu
@FileName: modelbuilder.py
@Software: PyCharm
'''

import json
import numpy as np
import os
import tensorflow as tf
import time
from train import G

os.environ['CUDA_VISIBLE_DEVICES'] = ''
# param
model_path = '/home/zhuqingjie/prj/tunet_onesample/model_release'
model_build_path = '/home/zhuqingjie/prj/tunet_onesample/SERVER/build_model'

model_ver = len(os.listdir(model_build_path)) + 1
print(f'model_ver:{model_ver}')
while True:
    if os.path.exists(os.path.join(model_build_path, str(model_ver))):
        model_ver += 1
    else:
        break

flist = os.listdir(model_path)
for f in flist:
    if "model_" in f:
        model_ind = f.split('.')[0]
        break
checkpoint_path = os.path.join(model_path, model_ind)

g = G(predict_flag=True)
with tf.Session(graph=g.graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    export_path = os.path.join(tf.compat.as_bytes(model_build_path), tf.compat.as_bytes(str(model_ver)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    inputs = g.x
    outputs = g.prd
    tensor_info_x = tf.saved_model.utils.build_tensor_info(inputs)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(outputs)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': tensor_info_x},
            outputs={"outputs": tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         {"predict_img": prediction_signature}, legacy_init_op=legacy_init_op)

    builder.save()

print()
