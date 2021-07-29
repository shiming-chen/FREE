#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shiming chen
"""
import os
os.system('''OMP_NUM_THREADS=8 python train_free.py --gammaD 10 \
--gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --nepoch 401 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 1 \
--feed_lr 0.0001 --dec_lr 0.0001 --loop 2 --a1 0.01 --a2 0.01 \
--nclass_all 50 --dataroot data --dataset AWA2 \
--batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
--lr 0.00001 --classifier_lr 0.001 --nclass_seen 40 --freeze_dec \
--syn_num 4600 --center_margin 50 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.5 ''')