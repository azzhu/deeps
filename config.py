#####################

#####################

# param
# 多gpu是坑，新版本将不支持多GPU
# epoch = 20
# batch_size = 8

prjdir = '/home/zhangli_lab/zhuqingjie/DATA/prj'

# data init
# 可以用的hw值：384,448,512
# size = hw = 256
# step = 80
data_path = f'{prjdir}/tunet_onesample/data/'

workdir = f'{prjdir}/tunet_onesample/'
# logdir = f'{prjdir}/tunet_onesample/logdir_nd2_justunet/'
logdir = f'{prjdir}/tunet_onesample/logdir_nd2_deeps/'

restore_model = False
restore_path = f'{prjdir}/tunet_onesample/logdir_nd2_justunet/model_099900'


def read_hot_config():
    lines = open('hot_config', 'r').readlines()
    return {l.split('=')[0].strip(): l.split('=')[1].strip() for l in lines if l.strip() and l[0] != '#'}


if __name__ == '__main__':
    hot_params = read_hot_config()
    print(hot_params)
