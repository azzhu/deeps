import tensorflow as tf
# import tensorflow.layers as L
from vgg16 import Vgg16 as VGG
import os, time
import config

L = tf.layers
print_ = lambda x: print(f"--> [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]: {x}")
printc = lambda s: print(f"\033[1;35m{s}\033[0m")
printc(tf.__version__)


def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# print(get_available_gpus)


# 统一框架，新版本统一使用UNET_sr框架
# 不要再使用这个，这个已被更改为UNET_sr
class UNET_os():
    def __init__(sf, predict_flag=False, H=None, W=None, learning_rate=0.0001):
        print_(f'TensorFlow version: {tf.__version__}')
        # print_(f'gpus: {get_available_gpus()}')
        print_('loading NetWork..')

        # param
        batch_size = config.batch_size
        batch_per_gpu = config.batch_per_gpu
        hw = config.size
        size = config.size
        gpus = config.gpus

        if predict_flag:
            gpus = gpus[0:1]

        # sf.readme = '''
        # Data: {}*{}
        # G: unet
        # D: cnn
        # Opt: adam, GAN loss + perc loss + MSE loss
        # Train: 又多train了2次train_op_G2
        # '''.format(size, size)
        sf.graph = tf.Graph()
        sf.istraining = not predict_flag

        # dataset_train, dataset_test = load_data_2()
        with sf.graph.as_default():
            if predict_flag:
                sf.x = tf.placeholder(tf.float32, [None, H, W, 1])
                # sf.istraining = True
                sf.prd = sf.gen(sf.x)
                return
            with tf.device('/cpu:0'):
                if sf.istraining:
                    sf.x = tf.placeholder(tf.float32, [None, H, W, 1])
                    sf.y = tf.placeholder(tf.float32, [None, H, W, 1])
                else:
                    sf.x = tf.placeholder(tf.float32, [None, H, W, 1])
                    sf.y = tf.placeholder(tf.float32, [None, H, W, 1])
                    prd = sf.gen(sf.x)
                    # abs_error = tf.reduce_mean(tf.abs(sf.y - prd)) * 255
                    # yr = tf.reshape(sf.y, [-1, hw * hw])
                    # prdr = tf.reshape(prd, [-1, hw * hw])
                    # loss_mse = tf.losses.mean_squared_error(yr, prdr)
                    # sf.loss_mse = loss_mse
                    # sf.abs_error = abs_error
                    # sf.x_ = sf.x
                    # sf.y_ = sf.y
                    sf.prd = prd
                    return

                # Multi GPU
                # sf.opt = tf.train.AdamOptimizer(0.0001)
                sf.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
                sf.global_step = tf.Variable(0, trainable=False)
                tower_grads_G = []
                tower_grads_D = []
                # with tf.variable_scope(tf.get_variable_scope()):    # 这一句真沙雕
                # 不管gpu_inds设置的是啥，给tensorflow识别后是从0开始index
                for gpu_i in range(len(gpus)):
                    with tf.device('/gpu:{}'.format(gpu_i)):
                        with tf.name_scope('tower_{}'.format(gpu_i)):
                            # split batch
                            x_ = sf.x[gpu_i * batch_per_gpu:(gpu_i + 1) * batch_per_gpu]
                            y_ = sf.y[gpu_i * batch_per_gpu:(gpu_i + 1) * batch_per_gpu]

                            # G and D
                            prd = sf.gen(x_)
                            d_prd = sf.disc(prd)

                            # loss
                            ls_d_prd = tf.reduce_mean(d_prd)
                            loss_d_prd = tf.reduce_mean(tf.log(tf.clip_by_value(d_prd, 1e-10, 1.0)))
                            d_y = sf.disc(y_)
                            ls_d_y = tf.reduce_mean(d_y)
                            loss_d_y = tf.reduce_mean(tf.log(tf.clip_by_value(d_y, 1e-10, 1.0)))
                            abs_error = tf.reduce_mean(tf.abs(y_ - prd)) * 255

                            # MSE Loss
                            new_dim = 1
                            for d in y_.shape[1:]:
                                new_dim *= d
                            yr = tf.reshape(y_, [-1, new_dim])
                            prdr = tf.reshape(prd, [-1, new_dim])
                            loss_mse = tf.losses.mean_squared_error(yr, prdr)

                            # Perceptual Loss
                            vgg = VGG()
                            y_perc = vgg.forward(y_)
                            prd_perc = vgg.forward(prd)
                            new_dim = 1
                            for d in y_perc.shape[1:]:
                                new_dim *= d
                            yr_perc = tf.reshape(y_perc, [-1, new_dim])
                            prdr_perc = tf.reshape(prd_perc, [-1, new_dim])
                            loss_perc = tf.losses.mean_squared_error(yr_perc, prdr_perc)

                            # Adversarial Loss
                            loss_G = loss_mse + loss_perc - loss_d_prd
                            loss_D = loss_d_prd - loss_d_y

                            # gard
                            var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                            var_list_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                            grads_G = sf.opt.compute_gradients(loss_G, var_list=var_list_G)
                            grads_D = sf.opt.compute_gradients(loss_D, var_list=var_list_D)
                            # grads_G2 = sf.opt.compute_gradients(-loss_d_prd, var_list=var_list_G)
                            tower_grads_G.append(grads_G)
                            tower_grads_D.append(grads_D)
                            # tower_grads_G2.append(grads_G2)

                            # summary
                            if gpu_i == 0:
                                sf.loss_mse = loss_mse
                                sf.loss_perc = loss_perc
                                sf.abs_error = abs_error
                                sf.loss_G = loss_G
                                sf.loss_D = loss_D
                                sf.ls_d_prd = ls_d_prd
                                sf.ls_d_y = ls_d_y
                                sf.loss_d_prd = loss_d_prd
                                sf.loss_d_y = loss_d_y
                                sf.x_ = x_
                                sf.y_ = y_
                                sf.prd = prd

                # summary
                tf.summary.scalar('loss1/loss', sf.loss_mse)
                tf.summary.scalar('loss1/loss_perc', sf.loss_perc)
                tf.summary.scalar('loss1/abs_error', sf.abs_error)
                tf.summary.scalar('DG_loss/G', sf.loss_G)
                tf.summary.scalar('DG_loss/D', sf.loss_D)
                tf.summary.scalar('Dloss/ls_d_prd', sf.ls_d_prd)
                tf.summary.scalar('Dloss/ls_d_y', sf.ls_d_y)
                tf.summary.scalar('Dloss/loss_d_prd', sf.loss_d_prd)
                tf.summary.scalar('Dloss/loss_d_y', sf.loss_d_y)
                tf.summary.image('img/x', sf.x_[0:1], max_outputs=1)
                tf.summary.image('img/y', sf.y_[0:1], max_outputs=1)
                tf.summary.image('img/prd', sf.prd[0:1], max_outputs=1)

                avg_grads_G = sf.average_gradients(tower_grads_G)
                avg_grads_D = sf.average_gradients(tower_grads_D)
                # avg_grads_G2 = sf.average_gradients(tower_grads_G2)
                sf.train_op_G = sf.opt.apply_gradients(avg_grads_G, global_step=sf.global_step)
                sf.train_op_D = sf.opt.apply_gradients(avg_grads_D)
                # sf.train_op_G2 = sf.opt.apply_gradients(avg_grads_G2)

                # tf.summary.text()
                sf.mergeall = tf.summary.merge_all()

    # generator
    def gen(sf, x):
        print_('loading generator..')
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            c1 = sf.conv(x, 64, 'c1')
            c2 = sf.conv(c1, 64, 'c2')
            p1 = L.max_pooling2d(c2, 2, 2, name='p1')
            c3 = sf.conv(p1, 128, 'c3')
            c4 = sf.conv(c3, 128, 'c4')
            p2 = L.max_pooling2d(c4, 2, 2, name='p2')
            c5 = sf.conv(p2, 256, 'c5')
            c6 = sf.conv(c5, 256, 'c6')
            p3 = L.max_pooling2d(c6, 2, 2, name='p3')
            c7 = sf.conv(p3, 512, 'c7')
            c8 = sf.conv(c7, 512, 'c8')
            d1 = L.dropout(c8, 0.5, training=sf.istraining, name='d1')
            p4 = L.max_pooling2d(d1, 2, 2, name='p4')
            c9 = sf.conv(p4, 1024, 'c9')
            c10 = sf.conv(c9, 1024, 'c10')
            d2 = L.dropout(c10, 0.5, training=sf.istraining, name='d2')

            u1 = tf.keras.layers.UpSampling2D()(d2)
            if u1.shape[1] != d1.shape[1]:
                u1 = tf.pad(u1, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u1.shape[2] != d1.shape[2]:
                u1 = tf.pad(u1, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc1 = sf.conv(u1, 512, 'uc1', ker_size=2)
            mg1 = tf.concat([d1, uc1], axis=3, name='mg1')
            c11 = sf.conv(mg1, 512, 'c11')
            c12 = sf.conv(c11, 512, 'c12')

            # u2 = L.conv2d_transpose(c12, 512, 3, 2, padding='same')
            u2 = tf.keras.layers.UpSampling2D()(c12)
            if u2.shape[1] != c6.shape[1]:
                u2 = tf.pad(u2, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u2.shape[2] != c6.shape[2]:
                u2 = tf.pad(u2, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc2 = sf.conv(u2, 256, 'uc2', ker_size=2)
            mg2 = tf.concat([c6, uc2], axis=3, name='mg2')
            c13 = sf.conv(mg2, 256, 'c13')
            c14 = sf.conv(c13, 256, 'c14')

            u3 = tf.keras.layers.UpSampling2D()(c14)
            if u3.shape[1] != c4.shape[1]:
                u3 = tf.pad(u3, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u3.shape[2] != c4.shape[2]:
                u3 = tf.pad(u3, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc3 = sf.conv(u3, 128, 'uc3', ker_size=2)
            mg3 = tf.concat([c4, uc3], axis=3, name='mg3')
            c15 = sf.conv(mg3, 128, 'c15')
            c16 = sf.conv(c15, 128, 'c16')

            # u4 = L.conv2d_transpose(c16, 128, 3, 2, padding='same')
            u4 = tf.keras.layers.UpSampling2D()(c16)
            if u4.shape[1] != c2.shape[1]:
                u4 = tf.pad(u4, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u4.shape[2] != c2.shape[2]:
                u4 = tf.pad(u4, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc4 = sf.conv(u4, 64, 'uc4', ker_size=2)
            mg4 = tf.concat([c2, uc4], axis=3, name='mg4')
            c17 = sf.conv(mg4, 64, 'c17')
            c18 = sf.conv(c17, 64, 'c18')
            c19 = sf.conv(c18, 2, 'c19')

            # tf.summary.histogram('c19', c19)
            prd = sf.conv(c19, 1, 'prd', ker_size=1, act=tf.nn.sigmoid)
            return prd

    def residual_unit_bottleneck(sf, input, channels=64, name=None, is_training=True):
        def conv_2d(inputs, channels, kernel_size=3, strides=1, batch_normalization=True,
                    padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    is_training=True, activation=True):
            output = tf.layers.conv2d(inputs=inputs, filters=channels, kernel_size=kernel_size, strides=strides,
                                      padding=padding, kernel_initializer=kernel_initializer)
            if batch_normalization:
                output = tf.layers.batch_normalization(output, axis=-1, momentum=0.9, training=is_training)
            if activation:
                output = tf.nn.leaky_relu(output)
            return output

        _, _, _, c = input.get_shape()
        conv_1x1_1 = conv_2d(input, channels=channels, kernel_size=1, is_training=is_training)
        conv_3x3 = conv_2d(conv_1x1_1, channels=channels, is_training=is_training)
        conv_1x1_2 = conv_2d(conv_3x3, channels=c, kernel_size=1, is_training=is_training, activation=False)
        _output = tf.add(input, conv_1x1_2)
        output = tf.nn.leaky_relu(_output)
        return output

    def gen_allcnn(sf, x):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            x = sf.conv(x, 16, 'c1', 3)
            x = sf.conv(x, 16, 'c2', 3)
            x = sf.conv(x, 32, 'c3', 7)
            x = sf.conv(x, 32, 'c4', 7)
            x = sf.conv(x, 64, 'c5', 15)
            x = sf.conv(x, 64, 'c6', 15)
            x = sf.conv(x, 32, 'c7', 7)
            x = sf.conv(x, 32, 'c8', 7)
            x = sf.conv(x, 16, 'c9', 3)
            x = sf.conv(x, 16, 'c10', 3)
            x = sf.conv(x, 1, 'c11', 1)
            return x

    def gen_resnet(sf, x):
        with tf.variable_scope('generator'):
            conv1 = L.conv2d(x, 64, 9, name='conv1', activation=tf.nn.leaky_relu, padding='same')
            rb1 = sf.res_block(conv1, 'rb1')
            rb2 = sf.res_block(rb1, 'rb2')
            rb3 = sf.res_block(rb2, 'rb3')
            rb4 = sf.res_block(rb3, 'rb4')
            rb5 = sf.res_block(rb4, 'rb5')
            conv2 = L.conv2d(rb5, 64, 3, name='conv2', padding='same')
            bn1 = L.batch_normalization(conv2, training=sf.istraining, name='bn1')
            sum1 = bn1 + conv1
            conv3 = L.conv2d(sum1, 256, 3, name='conv3', activation=tf.nn.leaky_relu, padding='same')
            conv4 = L.conv2d(conv3, 64, 3, name='conv4', activation=tf.nn.leaky_relu, padding='same')
            conv5 = L.conv2d(conv4, 1, 9, name='conv5', padding='same')
            return conv5

    def res_block(sf, x, name):
        with tf.variable_scope(name):
            conv1 = L.conv2d(x, 64, 3, name='conv1', padding='same')
            bn1 = L.batch_normalization(conv1, training=sf.istraining, name='bn1')
            act1 = tf.nn.leaky_relu(bn1)
            conv2 = L.conv2d(act1, 64, 3, name='conv2', padding='same')
            bn2 = L.batch_normalization(conv2, training=sf.istraining, name='bn2')
            return x + bn2

    def disc_resnet(sf, x):
        print_('loading discriminator..')
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            conv1 = L.conv2d(x, 16, 3, name='conv1', activation=tf.nn.leaky_relu)
            cb1 = sf.conv_block(conv1, 16, 'cb1', 2)
            cb2 = sf.conv_block(cb1, 32, 'cb2', 2)
            cb3 = sf.conv_block(cb2, 32, 'cb3', 1)
            cb4 = sf.conv_block(cb3, 64, 'cb4', 2)
            cb5 = sf.conv_block(cb4, 64, 'cb5', 2)
            cb6 = sf.conv_block(cb5, 128, 'cb6', 1)
            cb7 = sf.conv_block(cb6, 128, 'cb7', 2)
            line = tf.reshape(cb7, [cb7.shape[0], -1])
            fc1 = L.dense(line, 128, activation=tf.nn.leaky_relu, name='fc1')
            tf.summary.histogram('d_fc1', fc1)
            fc2 = L.dense(fc1, 1, activation=tf.nn.sigmoid, name='fc2')
            return fc2

    def conv_block(sf, x, n, name, s=1):
        with tf.variable_scope(name):
            conv1 = L.conv2d(x, n, 3, s, name='conv1')
            bn1 = L.batch_normalization(conv1, training=sf.istraining, name='bn1')
            act1 = tf.nn.leaky_relu(bn1)
            return act1

    # discriminator
    def disc(sf, prd):
        print_('loading discriminator..')
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            c1 = sf.conv(prd, 16, 'c1')
            # p1 = L.max_pooling2d(c1, 2, 2, name='p1')
            c2 = sf.conv(c1, 16, 'c2')
            p2 = L.max_pooling2d(c2, 2, 2, name='p2')
            c3 = sf.conv(p2, 32, 'c3')
            # p3 = L.max_pooling2d(c3, 2, 2, name='p3')
            c4 = sf.conv(c3, 32, 'c4')
            p4 = L.max_pooling2d(c4, 2, 2, name='p4')
            c5 = sf.conv(p4, 64, 'c5')
            p5 = L.max_pooling2d(c5, 2, 2, name='p5')
            c6 = sf.conv(p5, 64, 'c6')
            p6 = L.max_pooling2d(c6, 2, 2, name='p6')
            c7 = sf.conv(p6, 128, 'c7')
            p7 = L.max_pooling2d(c7, 2, 2, name='p7')
            c8 = sf.conv(p7, 128, 'c8', pad='valid')
            new_dim = 1
            for d in c8.shape[1:]:
                new_dim *= d
            line1 = tf.reshape(c8, [-1, new_dim])
            fc1 = L.dense(line1, 128, activation=tf.nn.leaky_relu)
            d1 = L.dropout(fc1, 0.5, training=sf.istraining)
            fc2 = L.dense(d1, 1, activation=tf.nn.sigmoid)
            return fc2

    def conv(sf, x, filters, name, ker_size=3,
             act=tf.nn.leaky_relu,
             pad='same',
             init=tf.contrib.layers.xavier_initializer()):
        return L.conv2d(x, filters, ker_size,
                        activation=act,
                        padding=pad,
                        kernel_initializer=init,
                        name=name)

    def average_gradients(sf, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


def vgg16(layer_ind=13):
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[layer_ind].output)
    model.trainable = False
    return model


class UNET_sr():
    def __init__(sf, predict_flag=False, H=None, W=None, learning_rate=0.0001):
        print_(f'TensorFlow version: {tf.__version__}')
        # print_(f'gpus: {get_available_gpus()}')
        print_('loading NetWork..')

        sf.graph = tf.Graph()
        sf.istraining = not predict_flag
        # dataset_train, dataset_test = load_data_2()
        with sf.graph.as_default():
            if predict_flag:
                sf.x = tf.placeholder(tf.float32, [None, H, W, 1])
                # sf.istraining = True
                sf.prd = sf.gen(sf.x)
                return

            # feed mode
            sf.x = tf.placeholder(tf.float32, [None, H, W, 1])
            sf.y = tf.placeholder(tf.float32, [None, H, W, 1])

            sf.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            sf.global_step = tf.Variable(0, trainable=False)

            x_ = sf.x
            y_ = sf.y

            prd = sf.gen(x_)
            # d_prd = sf.disc(prd)
            # loss_d_prd = tf.reduce_mean(d_prd)
            # loss_d_prd = tf.reduce_mean(tf.log(tf.clip_by_value(d_prd, 1e-10, 1.0)))
            # d_y = sf.disc(y_)
            # loss_d_y = tf.reduce_mean(d_y)
            # loss_d_y = tf.reduce_mean(tf.log(tf.clip_by_value(d_y, 1e-10, 1.0)))
            abs_error = tf.reduce_mean(tf.abs(y_ - prd))
            sf.abs_error = abs_error

            # MSE Loss
            # yr = tf.reshape(y_, [y_.shape[0], -1])
            # prdr = tf.reshape(prd, [prd.shape[0], -1])
            # loss_mse = tf.losses.mean_squared_error(yr, prdr)
            loss_mse = tf.keras.losses.MeanSquaredError()(y_, prd)
            # loss_ce = tf.losses.sigmoid_cross_entropy(yr, prdr)
            # loss_bc = tf.keras.losses.BinaryCrossentropy()(y_, prd)

            # Perceptual Loss
            # vgg = VGG()
            # y_perc = vgg.forward(y_)
            # prd_perc = vgg.forward(prd)
            # vgg = vgg16()
            # y_perc = vgg(tf.concat([y_] * 3, axis=-1))
            # prd_perc = vgg(tf.concat([prd] * 3, axis=-1))
            # yr_perc = tf.reshape(y_perc, [y_perc.shape[0], -1])
            # prdr_perc = tf.reshape(prd_perc, [prd_perc.shape[0], -1])
            # loss_perc = tf.losses.mean_squared_error(yr_perc, prdr_perc)
            # loss_perc = tf.reduce_mean(tf.abs(y_perc - prd_perc))

            # Adversarial Loss
            # loss_G = abs_error + loss_perc - loss_d_prd
            # loss_G = loss_mse * 1e2 + loss_perc * 1e-6 - loss_d_prd
            loss_G = loss_mse
            # loss_D = (loss_d_prd - loss_d_y) * 10

            # gard
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                # var_list_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                grads_G = sf.opt.compute_gradients(loss_G, var_list=var_list_G)
                # grads_D = sf.opt.compute_gradients(loss_D, var_list=var_list_D)
                sf.train_op_G = sf.opt.apply_gradients(grads_G, global_step=sf.global_step)
                # sf.train_op_D = sf.opt.apply_gradients(grads_D)

            # summary
            tf.summary.scalar('loss1/loss_mse', loss_mse)
            # tf.summary.scalar('loss1/loss_bc', loss_bc)
            # tf.summary.scalar('loss1/loss_perc', loss_perc)
            tf.summary.scalar('loss1/abs_error', abs_error * 255)
            # tf.summary.scalar('DG_loss/G', loss_G)
            # tf.summary.scalar('DG_loss/D', loss_D)
            # tf.summary.scalar('Dloss/loss_d_prd', loss_d_prd)
            # tf.summary.scalar('Dloss/loss_d_y', loss_d_y)
            tf.summary.image('img/0x', x_[0:1], max_outputs=1)
            tf.summary.image('img/2y', y_[0:1], max_outputs=1)
            tf.summary.image('img/1prd', prd[0:1], max_outputs=1)
            tf.summary.histogram('prd', prd)
            tf.summary.histogram('gt', y_)
            sf.mergeall = tf.summary.merge_all()

    # generator
    def gen(sf, x):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            c1 = sf.conv(x, 64, 'c1')
            c2 = sf.residual_unit_bottleneck(c1, is_training=sf.istraining)
            p1 = L.max_pooling2d(c2, 2, 2, name='p1')
            c3 = sf.conv(p1, 128, 'c3')
            c4 = sf.residual_unit_bottleneck(c3, is_training=sf.istraining)
            p2 = L.max_pooling2d(c4, 2, 2, name='p2')
            c5 = sf.conv(p2, 256, 'c5')
            c6 = sf.residual_unit_bottleneck(c5, is_training=sf.istraining)
            p3 = L.max_pooling2d(c6, 2, 2, name='p3')
            c7 = sf.conv(p3, 512, 'c7')
            c8 = sf.residual_unit_bottleneck(c7, is_training=sf.istraining)
            d1 = L.dropout(c8, 0.5, training=sf.istraining, name='d1')
            p4 = L.max_pooling2d(d1, 2, 2, name='p4')
            c9 = sf.conv(p4, 1024, 'c9')
            c10 = sf.residual_unit_bottleneck(c9, is_training=sf.istraining)
            d2 = L.dropout(c10, 0.5, training=sf.istraining, name='d2')

            u1 = tf.keras.layers.UpSampling2D()(d2)
            if u1.shape[1] != d1.shape[1]:
                u1 = tf.pad(u1, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u1.shape[2] != d1.shape[2]:
                u1 = tf.pad(u1, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc1 = sf.conv(u1, 512, 'uc1', ker_size=2)
            mg1 = tf.concat([d1, uc1], axis=3, name='mg1')
            c11 = sf.conv(mg1, 512, 'c11')
            c12 = sf.residual_unit_bottleneck(c11, is_training=sf.istraining)

            u2 = tf.keras.layers.UpSampling2D()(c12)
            if u2.shape[1] != c6.shape[1]:
                u2 = tf.pad(u2, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u2.shape[2] != c6.shape[2]:
                u2 = tf.pad(u2, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc2 = sf.conv(u2, 256, 'uc2', ker_size=2)
            mg2 = tf.concat([c6, uc2], axis=3, name='mg2')
            c13 = sf.conv(mg2, 256, 'c13')
            c14 = sf.residual_unit_bottleneck(c13, is_training=sf.istraining)

            u3 = tf.keras.layers.UpSampling2D()(c14)
            if u3.shape[1] != c4.shape[1]:
                u3 = tf.pad(u3, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u3.shape[2] != c4.shape[2]:
                u3 = tf.pad(u3, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc3 = sf.conv(u3, 128, 'uc3', ker_size=2)
            mg3 = tf.concat([c4, uc3], axis=3, name='mg3')
            c15 = sf.conv(mg3, 128, 'c15')
            c16 = sf.residual_unit_bottleneck(c15, is_training=sf.istraining)

            u4 = tf.keras.layers.UpSampling2D()(c16)
            if u4.shape[1] != c2.shape[1]:
                u4 = tf.pad(u4, tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), 'SYMMETRIC')
            if u4.shape[2] != c2.shape[2]:
                u4 = tf.pad(u4, tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
            uc4 = sf.conv(u4, 64, 'uc4', ker_size=2)
            mg4 = tf.concat([c2, uc4], axis=3, name='mg4')
            c17 = sf.conv(mg4, 64, 'c17')
            c18 = sf.residual_unit_bottleneck(c17, is_training=sf.istraining)

            c19 = sf.conv(c18, 2, 'c19')

            # tf.summary.histogram('c19', c19)
            prd = sf.conv(c19, 1, 'prd', ker_size=1, act=tf.nn.sigmoid)
            # prd = L.conv2d(c19, 1, 3, activation=None, padding='same', name='prd')
            return prd

    def residual_unit_bottleneck(sf, input, channels=64, name=None, is_training=True):
        def conv_2d(inputs, channels, kernel_size=3, strides=1, batch_normalization=True,
                    padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    is_training=True, activation=True):
            output = tf.layers.conv2d(inputs=inputs, filters=channels, kernel_size=kernel_size, strides=strides,
                                      padding=padding, kernel_initializer=kernel_initializer)
            if batch_normalization:
                output = tf.layers.batch_normalization(output, axis=-1, momentum=0.9, training=is_training)
            if activation:
                output = tf.nn.leaky_relu(output)
            return output

        _, _, _, c = input.get_shape()
        conv_1x1_1 = conv_2d(input, channels=channels, kernel_size=1, is_training=is_training)
        conv_3x3 = conv_2d(conv_1x1_1, channels=channels, is_training=is_training)
        conv_1x1_2 = conv_2d(conv_3x3, channels=c, kernel_size=1, is_training=is_training, activation=False)
        _output = tf.add(input, conv_1x1_2)
        output = tf.nn.leaky_relu(_output)
        return output

    def gen_allcnn(sf, x):
        with tf.variable_scope('generator'):
            x = sf.conv(x, 16, 'c1', 3)
            x = sf.conv(x, 16, 'c2', 3)
            x = sf.conv(x, 32, 'c3', 7)
            x = sf.conv(x, 32, 'c4', 7)
            x = sf.conv(x, 64, 'c5', 15)
            x = sf.conv(x, 64, 'c6', 15)
            # x = sf.conv(x, 128, 'c51', 30)
            # x = sf.conv(x, 128, 'c61', 30)
            # x = sf.conv(x, 64, 'c52', 15)
            # x = sf.conv(x, 64, 'c62', 15)
            x = sf.conv(x, 32, 'c7', 7)
            x = sf.conv(x, 32, 'c8', 7)
            x = sf.conv(x, 16, 'c9', 3)
            x = sf.conv(x, 16, 'c10', 3)
            x = sf.conv(x, 1, 'c11', 1)
            return x

    def gen_resnet(sf, x):
        with tf.variable_scope('generator'):
            conv1 = L.conv2d(x, 64, 9, name='conv1', activation=tf.nn.leaky_relu, padding='same')
            rb1 = sf.res_block(conv1, 'rb1')
            rb2 = sf.res_block(rb1, 'rb2')
            rb3 = sf.res_block(rb2, 'rb3')
            rb4 = sf.res_block(rb3, 'rb4')
            rb5 = sf.res_block(rb4, 'rb5')
            conv2 = L.conv2d(rb5, 64, 3, name='conv2', padding='same')
            bn1 = L.batch_normalization(conv2, training=sf.istraining, name='bn1')
            sum1 = bn1 + conv1
            conv3 = L.conv2d(sum1, 256, 3, name='conv3', activation=tf.nn.leaky_relu, padding='same')
            conv4 = L.conv2d(conv3, 64, 3, name='conv4', activation=tf.nn.leaky_relu, padding='same')
            conv5 = L.conv2d(conv4, 1, 9, name='conv5', padding='same')
            return conv5

    def res_block(sf, x, name):
        with tf.variable_scope(name):
            conv1 = L.conv2d(x, 64, 3, name='conv1', padding='same')
            bn1 = L.batch_normalization(conv1, training=sf.istraining, name='bn1')
            act1 = tf.nn.leaky_relu(bn1)
            conv2 = L.conv2d(act1, 64, 3, name='conv2', padding='same')
            bn2 = L.batch_normalization(conv2, training=sf.istraining, name='bn2')
            return x + bn2

    def disc_resnet(sf, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            conv1 = L.conv2d(x, 16, 3, name='conv1', activation=tf.nn.leaky_relu)
            cb1 = sf.conv_block(conv1, 16, 'cb1', 2)
            cb2 = sf.conv_block(cb1, 32, 'cb2', 2)
            cb3 = sf.conv_block(cb2, 32, 'cb3', 1)
            cb4 = sf.conv_block(cb3, 64, 'cb4', 2)
            cb5 = sf.conv_block(cb4, 64, 'cb5', 2)
            cb6 = sf.conv_block(cb5, 128, 'cb6', 1)
            cb7 = sf.conv_block(cb6, 128, 'cb7', 2)
            line = tf.reshape(cb7, [cb7.shape[0], -1])
            fc1 = L.dense(line, 128, activation=tf.nn.leaky_relu, name='fc1')
            tf.summary.histogram('d_fc1', fc1)
            fc2 = L.dense(fc1, 1, activation=tf.nn.sigmoid, name='fc2')
            return fc2

    def conv_block(sf, x, n, name, s=1):
        with tf.variable_scope(name):
            conv1 = L.conv2d(x, n, 3, s, name='conv1')
            bn1 = L.batch_normalization(conv1, training=sf.istraining, name='bn1')
            act1 = tf.nn.leaky_relu(bn1)
            return act1

    # discriminator
    def disc(sf, prd):
        print_('loading discriminator..')
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            c1 = sf.conv(prd, 16, 'c1')
            # p1 = L.max_pooling2d(c1, 2, 2, name='p1')
            c2 = sf.conv(c1, 16, 'c2')
            p2 = L.max_pooling2d(c2, 2, 2, name='p2')
            c3 = sf.conv(p2, 32, 'c3')
            # p3 = L.max_pooling2d(c3, 2, 2, name='p3')
            c4 = sf.conv(c3, 32, 'c4')
            p4 = L.max_pooling2d(c4, 2, 2, name='p4')
            c5 = sf.conv(p4, 64, 'c5')
            p5 = L.max_pooling2d(c5, 2, 2, name='p5')
            c6 = sf.conv(p5, 64, 'c6')
            p6 = L.max_pooling2d(c6, 2, 2, name='p6')
            c7 = sf.conv(p6, 128, 'c7')
            p7 = L.max_pooling2d(c7, 2, 2, name='p7')
            c8 = sf.conv(p7, 128, 'c8', pad='valid')
            new_dim = 1
            for d in c8.shape[1:]:
                new_dim *= d
            line1 = tf.reshape(c8, [-1, new_dim])
            fc1 = L.dense(line1, 128, activation=tf.nn.leaky_relu)
            d1 = L.dropout(fc1, 0.5, training=sf.istraining)
            fc2 = L.dense(d1, 1, activation=tf.nn.sigmoid)
            return fc2

    def conv(sf, x, filters, name, ker_size=3,
             act=tf.nn.leaky_relu,
             pad='same',
             init=tf.contrib.layers.xavier_initializer()):
        return L.conv2d(x, filters, ker_size,
                        activation=act,
                        padding=pad,
                        kernel_initializer=init,
                        name=name)

    def get_xy(sf, train_flag):
        def fn1():
            xy_dict = sf.train_iterator.get_next()
            return xy_dict[0], xy_dict[1]

        def fn2():
            xy_dict = sf.test_iterator.get_next()
            return xy_dict[0], xy_dict[1]

        return tf.cond(train_flag, fn1, fn2)

    def average_gradients(sf, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


# 统一框架，新版本统一使用UNET_sr框架
UNET_os = UNET_sr

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # x = tf.placeholder(tf.float32, [32, 512, 512, 1])
    # c1 = L.conv2d(x, 16, 3, padding='same')
    # p1 = L.max_pooling2d(c1, 2, 2)
    # tc1 = L.conv2d_transpose(p1, 16, 3, 2, padding='same')
    # exit()
    # raise ValueError('depth_multiplier is not greater than zero.')

    g = UNET_sr(predict_flag=False, H=512, W=512)
    with tf.Session(graph=g.graph) as sess:
        # var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        # saver = tf.train.Saver(var_list=var_list_G)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # prd = sess.run([g.prd], feed_dict={g.x: img})

    # [print(n.name) for n in net.graph.as_graph_def().node]
