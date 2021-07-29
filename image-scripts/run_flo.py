#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
os.system('''OMP_NUM_THREADS=8 python train_free.py \
--gammaD 10 --gammaG 10 --gzsl --nclass_all 102 --latent_size 1024 --manualSeed 806 \
--preprocessing --class_embedding att --nepoch 501 --ngh 4096 \
--loop 2 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --dec_lr 0.0001 \
--ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 256 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 \
--classifier_lr 0.001 --cuda --image_embedding res101 --dataroot data --nclass_seen 82\
--syn_num 2400 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.8 ''')