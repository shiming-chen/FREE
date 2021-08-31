
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Encoder
class Encoder(nn.Module):

    def __init__(self, opt):

        super(Encoder,self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

#Decoder/Generator
class Generator(nn.Module):

    def __init__(self, opt):

        super(Generator,self).__init__()

        layer_sizes = opt.decoder_layer_sizes
        latent_size=opt.latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid=nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x
       

#conditional discriminator
class Discriminator(nn.Module):
    def __init__(self, opt): 
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h
        






class FR(nn.Module):
    def __init__(self, opt, attSize):
        super(FR, self).__init__()
        self.embedSz = 0
        self.hidden = None
        self.lantent = None
        self.latensize=opt.latensize
        self.attSize = opt.attSize
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize*2)
        # self.encoder_linear = nn.Linear(opt.resSize, opt.latensize*2)
        self.discriminator = nn.Linear(opt.attSize, 1)
        self.classifier = nn.Linear(opt.attSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, feat, train_G=False):
        h = feat
        #if self.embedSz > 0:
        #   assert att is not None, 'Conditional Decoder requires attribute input'
        #    h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        self.lantent = self.fc3(self.hidden)
        mus,stds = self.lantent[:,:self.attSize],self.lantent[:,self.attSize:]
        stds=self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        h= encoder_out
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred=self.logic(self.classifier(mus))
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0),h.size(1))
        return mus, stds, dis_out, pred, encoder_out, h
        
    def getLayersOutDet(self):
        #used at synthesis time and feature transformation
        return self.hidden.detach()

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu
