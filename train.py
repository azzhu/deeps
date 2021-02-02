import tensorflow as tf
import os, time
import random
import numpy as np
# from data_init import get_alldatapath_big, load_datas_npy, batch_datas_random, load_labelimg
from model import UNET_sr as G
import config
from data_init import Datas_nd2
from pathlib import Path

# param
STEPS = int(1e5)


def train():
    data = Datas_nd2()

    g = G(predict_flag=False, H=256, W=256)

    # # launch tensorboard
    # os.system('/usr/sbin/fuser -k -n tcp 5005')
    # # os.system(f'rm {config.logdir}/checkpoint')
    # os.system(f'rm {config.logdir}/event*')
    # os.system(f'rm {config.logdir}/model_*')
    # os.system(f'rm {config.logdir}/v/event*')
    # time.sleep(1)
    # os.system(f'nohup /home/zhuqingjie/env/py3_tf_low/bin/tensorboard --logdir={config.logdir} --port=5005&')
    #
    # # 备份一份model.py
    # os.system(f"cp -p {config.workdir}model.py {config.logdir}/")

    # train
    with tf.Session(graph=g.graph) as sess:
        var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        bn_moving_vars = [g for g in bn_moving_vars if 'generator' in g.name]
        saver = tf.train.Saver(max_to_keep=1, var_list=var_list_G + bn_moving_vars)
        summary_writer = tf.summary.FileWriter(logdir=config.logdir)
        Path(config.logdir, 'v').mkdir(exist_ok=True)
        summary_writer_v = tf.summary.FileWriter(logdir=os.path.join(config.logdir, 'v'))
        sess.run(tf.global_variables_initializer())
        if config.restore_model:
            print('restore_model')
            saver_temp = tf.train.Saver(var_list=var_list_G)
            saver_temp.restore(sess, config.restore_path)

        time_use = []
        for step in range(STEPS):
            time_start = time.time()
            xs, ys = data.get_batch()
            _, _, summary, abs_error, gs = sess.run(
                [g.train_op_G, g.train_op_D, g.mergeall, g.abs_error, g.global_step],
                feed_dict={g.x: xs, g.y: ys})
            # _, summary, abs_error, gs = sess.run(
            #     [g.train_op_G, g.mergeall, g.abs_error, g.global_step],
            #     feed_dict={g.x: xs, g.y: ys})
            time_end = time.time()
            time_use.append(time_end - time_start)
            summary_writer.add_summary(summary, gs)
            print(f'{step}/{STEPS} abs_error:{abs_error * 255:.2f}')

            # val
            if gs % 10 == 0:
                xs, ys = data.get_batch_test()
                summary_, abs_error_ = sess.run(
                    [g.mergeall, g.abs_error],
                    feed_dict={g.x: xs, g.y: ys})
                summary_writer_v.add_summary(summary_, gs)
                print(f'-----------------------------------'
                      f'avg_time_use: {np.mean(np.array(time_use)):.3f}'
                      f'-----------------------------------')
                time_use = []

            # save
            if gs % 300 == 0:
                print(f'saved: {config.logdir}/model_{gs:0>6d}')
                saver.save(sess, f'{config.logdir}/model_{gs:0>6d}')


if __name__ == '__main__':
    print('\n' * 2)
    print('=' * 150)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('=' * 150)
    train()
    print('ok')
