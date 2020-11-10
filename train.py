import tensorflow as tf
import os, time
import random
import numpy as np
# from data_init import get_alldatapath_big, load_datas_npy, batch_datas_random, load_labelimg
from model import UNET as G
import config

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpus))
os.system('echo $CUDA_VISIBLE_DEVICES')

# param
epochs = config.epoch


class BreakAll(Exception):
    pass


def train():
    # img图像是从1开始的，而不是0
    labelimgs = load_labelimg()
    labelimgs_ = load_labelimg()

    g = G()

    # launch tensorboard
    os.system('/usr/sbin/fuser -k -n tcp 5005')
    # os.system(f'rm {config.logdir}/checkpoint')
    os.system(f'rm {config.logdir}/event*')
    os.system(f'rm {config.logdir}/model_*')
    os.system(f'rm {config.logdir}/v/event*')
    time.sleep(1)
    os.system(f'nohup /home/zhuqingjie/env/py3_tf_low/bin/tensorboard --logdir={config.logdir} --port=5005&')

    # 备份一份model.py
    os.system(f"cp -p {config.workdir}model.py {config.logdir}/")

    # train
    with tf.Session(graph=g.graph) as sess:
        var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        saver = tf.train.Saver(max_to_keep=1, var_list=var_list_G)
        # saver = tf.train.Saver(max_to_keep=1)
        summary_writer = tf.summary.FileWriter(logdir=config.logdir)
        summary_writer_v = tf.summary.FileWriter(logdir=os.path.join(config.logdir, 'v'))
        if config.restore_model:
            print('restore_model')
            print(f'{config.restore_path}last_random_ind.txt')
            files_t_x, files_t_y, files_v_x, files_v_y = \
                get_alldatapath_big()
            flist = os.listdir(config.restore_path)
            for f in flist:
                if "model_" in f:
                    model_ind = f.split('.')[0]
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, os.path.join(config.restore_path, model_ind))
                    break
        else:
            files_t_x, files_t_y, files_v_x, files_v_y = get_alldatapath_big()
            sess.run(tf.global_variables_initializer())
        try:
            time_use = []
            time_use2 = []
            begin_ind = 0
            while True:
                xt_batchs, yt_batchs, xv_batchs, yv_batchs = \
                    batch_datas_random(files_t_x, files_t_y, files_v_x, files_v_y)
                for batch_xs, batch_ys in zip(xt_batchs, yt_batchs):
                    time_start = time.time()
                    xs, ys, x_zs, inds = load_datas_npy(batch_xs, batch_ys)
                    lbimgs = np.array([labelimgs[lbi] for lbi in inds])
                    time_end = time.time()
                    hot_params = config.read_hot_config()
                    assert 'learning_rate' in hot_params and 'prec_loss_w' in hot_params, \
                        'learning_rate or prec_loss_w not in hot_config'
                    _, _, summary, abs_error, gs = sess.run(
                        [g.train_op_G, g.train_op_D, g.mergeall, g.abs_error, g.global_step],
                        feed_dict={g.x: xs, g.y: ys, g.x_z: x_zs, g.labelimg: lbimgs,
                                   g.learning_rate: float(hot_params['learning_rate']),
                                   g.loss_prec_w: float(hot_params['prec_loss_w']),
                                   g.loss_d_w: float(hot_params['loss_d_w']),
                                   g.loss_mse_w: float(hot_params['loss_mse_w']),
                                   g.loss_ms_w: float(hot_params['loss_ms_w']),
                                   })
                    time_end2 = time.time()
                    time_use.append(time_end - time_start)
                    time_use2.append(time_end2 - time_start)
                    summary_writer.add_summary(summary, gs)
                    # val
                    if gs % 10 == 0:
                        index = random.randint(0, len(xv_batchs) - 1)
                        xs_, ys_, x_zs_, inds_ = load_datas_npy(xv_batchs[index], yv_batchs[index])
                        lbimgs_ = np.array([labelimgs_[lbi] for lbi in inds_])
                        summary_, gs_ = sess.run(
                            [g.mergeall, g.global_step],
                            feed_dict={g.x: xs_, g.y: ys_, g.x_z: x_zs_, g.labelimg: lbimgs_,
                                       g.learning_rate: float(hot_params['learning_rate']),
                                       g.loss_prec_w: float(hot_params['prec_loss_w']),
                                       g.loss_d_w: float(hot_params['loss_d_w']),
                                       g.loss_mse_w: float(hot_params['loss_mse_w']),
                                       g.loss_ms_w: float(hot_params['loss_ms_w']),
                                       })
                        summary_writer_v.add_summary(summary_, gs_)
                        print(f'---------------'
                              f'avg_time_use:{np.mean(np.array(time_use)):.3f} {np.mean(np.array(time_use2)):.3f}')
                        print('gs / data_len, gs, loss, abs_error')
                        time_use = []
                        time_use2 = []
                    # save
                    if gs % 100 == 0:
                        saver.save(sess, f'{config.logdir}/model_{gs}')
                    begin_ind += 1
                    ep = begin_ind / len(xt_batchs)
                    print(f'{ep:.2f} -- {gs} --  [params:learning_rate,{hot_params["learning_rate"]};prec_loss_w,'
                          f'{hot_params["prec_loss_w"]}] -- {abs_error:.4f}')
                    if ep > epochs: raise BreakAll
        except BreakAll:
            print('BreakAll')
            pass


if __name__ == '__main__':
    print('\n' * 2)
    print('=' * 150)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('=' * 150)
    train()
    print('ok')
