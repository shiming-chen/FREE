#######################
#author: Shiming Chen
#FREE
#######################
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
import csv
#import functions
import model
import util
import classifier as classifier_zero

from config import opt
import time
import classifier_cls as classifier2
from center_loss import TripCenterLoss_min_margin,TripCenterLoss_margin


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

cls_criterion = nn.NLLLoss()
if opt.dataset in ['CUB','FLO']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
elif opt.dataset in ['AWA1','AWA2', 'APY']:
    center_criterion = TripCenterLoss_min_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
elif opt.dataset in ['SUN']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
else:
    raise ValueError('Dataset %s is not supported'%(opt.dataset))

netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator(opt)
# Init models: Feedback module, auxillary module
 
netFR = model.FR(opt, opt.attSize)

print(netE)
print(netG)
print(netD)
print(netFR)
 
###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize) #attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)
#one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
beta=0
##########
# Cuda
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netG.cuda()

    netFR.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    input_label=input_label.cuda()

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    #return (KLD)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
    
def generate_syn_feature(generator,classes, attribute,num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        fake = generator(syn_noisev,c=syn_attv)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerFR      = optim.Adam(netFR.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizer_center   = optim.Adam(center_criterion.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def calc_gradient_penalty_FR(netFR, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    _,_,disc_interpolates,_ ,_, _ = netFR(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                  - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

    MI_loss = (torch.mean(kl_divergence) - i_c)

    return MI_loss

def optimize_beta(beta, MI_loss,alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))

    # return the updated beta value:
    return beta_new



if not os.path.exists(os.path.join(opt.result_root, opt.dataset)):
    os.makedirs(os.path.join(opt.result_root, opt.dataset))

best_gzsl_acc = 0
best_zsl_acc = 0
for epoch in range(0,opt.nepoch):
    for loop in range(0,opt.loop):
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, data.ntrain, opt.batch_size):
            #########Discriminator training ##############
            for p in netD.parameters(): #unfreeze discrimator
                p.requires_grad = True

            for p in netFR.parameters(): #unfreeze deocder
                p.requires_grad = True
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 #lAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()          
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                
                if opt.encoded_noise:        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means #torch.Size([64, 312])
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)
                
                ################# update FR
                netFR.zero_grad()
                muR, varR, criticD_real_FR, latent_pred, _, recons_real = netFR(input_resv)
                criticD_real_FR = criticD_real_FR.mean()
                R_cost = opt.recons_weight*WeightedL1(recons_real, input_attv) 
                
                fake = netG(z, c=input_attv)
                muF, varF, criticD_fake_FR, _, _, recons_fake= netFR(fake.detach())
                criticD_fake_FR = criticD_fake_FR.mean()
                gradient_penalty = calc_gradient_penalty_FR(netFR, input_resv, fake.data)
                center_loss_real=center_criterion(muR, input_label,margin=opt.center_margin, incenter_weight=opt.incenter_weight)
                D_cost_FR = center_loss_real*opt.center_weight + R_cost
                D_cost_FR.backward()
                optimizerFR.step()
                optimizer_center.step()
                
                ############################
                
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)
                
                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)

                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae 
                optimizerD.step()
                
                
                
                
                
                
                

            gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discrimator
                p.requires_grad = False
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netFR.parameters(): #freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means #torch.Size([64, 312])

            recon_x = netG(z, c=input_attv)
            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var) # minimize E 3 with this setting feedback will update the loss as well
            errG = vae_loss_seen
            
            if opt.encoded_noise:
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)

                fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake,input_attv).mean()
                

            G_cost = -criticG_fake
            errG += opt.gammaG*G_cost
            
            ######################################original
            netFR.zero_grad()
            _,_,criticG_fake_FR,latent_pred_fake, _, recons_fake = netFR(fake, train_G=True)
            R_cost = WeightedL1(recons_fake, input_attv)
            errG += opt.recons_weight * R_cost
            

            
            errG.backward()
            # write a condition here
            optimizer.step()
            optimizerG.step()
            # if opt.recons_weight > 0 and not opt.freeze_dec: # not train decoder at feedback time
            optimizerFR.step() 
        
    
        
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),vae_loss_seen.item()))#,end=" ")
    netG.eval()
    netFR.eval()

    syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num)

    ### Concatenate real seen features with synthesized unseen features
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    nclass = opt.nclass_all
    ### Train GZSL classifier
    gzsl_cls = classifier_zero.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5,25, opt.syn_num, netFR=netFR, dec_size=opt.attSize, dec_hidden_size=(opt.latensize*2))
    
    if best_gzsl_acc <= gzsl_cls.H:
        best_gzsl_epoch= epoch
        best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
        ### torch.save({'netG': netG.state_dict()}, os.path.join(opt.result_root, opt.dataset, 'checkpoint_G.pth.tar'))
        ### torch.save({'netFR': netFR.state_dict()}, os.path.join(opt.result_root, opt.dataset, 'checkpoint_F.pth.tar'))
    print('GZSL: seen=%.3f, unseen=%.3f, h=%.3f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")
        
    if epoch % 10 == 0:
        print('GZSL: epoch=%d, best_seen=%.3f, best_unseen=%.3f, best_h=%.3f' % (best_gzsl_epoch, best_acc_seen, best_acc_unseen, best_gzsl_acc))
    # print('ZSL: epoch=%d, best unseen accuracy=%.4f' % (best_zsl_epoch, best_zsl_acc))
    
    
    # reset G to training mode
    netG.train()
    netFR.train()

# print('feature(X+feat1): 2048+4096')
print('softmax: feature(X+feat1+feat2): 8494')
print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)
