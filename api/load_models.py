import torch 
from torch import nn as nn 
from collections import OrderedDict
import pickle, os
import numpy as np 
from style_gan_model import G_mapping, G_synthesis
from dcgan_model import Generator

def load_stylegan2(): 
    with open('style_gan2.pkl', 'rb') as f:
        stylegan2_generator = pickle.load(f)#['G_ema']   
    stylegan2_generator = stylegan2_generator.float() # convert to float
    return stylegan2_generator

def load_stylegan(): 
    stylegan_generator = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        ('g_synthesis', G_synthesis())
    ]))

    stylegan_generator.load_state_dict(torch.load(
    './karras2019stylegan-ffhq-1024x1024.for_g_all.pt'))

    return stylegan_generator

def load_dcgan(): 
    models = torch.load("dcgan_model.pth", map_location=torch.device('cpu'))
    params = {
        "bsize": 128,  
        'imsize': 64,
        'nc': 3,
        'nz': 100, 
        'ngf': 64,
        'ndf': 64,
        'nepochs': 10,  
        'lr': 0.0002, 
        'beta1': 0.5, 
        'save_epoch': 2}  

    dcgan_generator = Generator(params)
    dcgan_generator.load_state_dict(models['generator'])
    return dcgan_generator
