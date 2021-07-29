#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shiming chen
"""
import os
os.system('''OMP_NUM_THREADS=8  python train_free.py --gammaD 10 --gammaG 10 \
--gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
--nepoch 501 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot data --dataset CUB \
 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
--nclass_all 200 --nclass_seen 150 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048  \
--syn_num 700 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.8''')