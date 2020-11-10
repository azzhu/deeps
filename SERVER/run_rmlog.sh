#!/usr/bin/bash

module load cuda/10.0
module load cudnn/7.4.2
source /home/zhangli_lab/zhuqingjie/act_PY3_TF1

rm log/log_?
rm log/log_??
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_1.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_2.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_21.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_22.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_3.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_4.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_5.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_51.ini
uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_6.ini


#uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_10.ini
#uwsgi /home/zhangli_lab/zhuqingjie/DATA/prj/tunet_onesample/SERVER/uwsgi_config/uwsgi_control.ini
#python control_sk.py