from flask import Flask, send_file, request
import imageio
import torch
import numpy as np
from style_gan_model import G_mapping, G_synthesis
from dcgan_model import Generator
import torch
from torch import nn as nn
from collections import OrderedDict
import os, pickle 

# so imageio can stfu
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# StyleGAN 2 Model 
with open('style_gan2.pkl', 'rb') as f:
    stylegan2_generator = pickle.load(f)#['G_ema']   
stylegan2_generator = stylegan2_generator.float() # convert to float

# StyleGAN model

stylegan_generator = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    ('g_synthesis', G_synthesis())
]))

stylegan_generator.load_state_dict(torch.load(
    './karras2019stylegan-ffhq-1024x1024.for_g_all.pt'))

# DCGAN model
models = torch.load("dcgan_model.pth", map_location=torch.device('cpu'))
params = {
    "bsize": 128,  # Batch size during training.
    # Spatial size of training images. All images will be resized to this size during preprocessing.
    'imsize': 64,
    # Number of channles in the training images. For coloured images this is 3.
    'nc': 3,
    'nz': 100,  # Size of the Z latent vector (the input to the generator).
    # Size of feature maps in the generator. The depth will be multiples of this.
    'ngf': 64,
    # Size of features maps in the discriminator. The depth will be multiples of this.
    'ndf': 64,
    'nepochs': 10,  # Number of training epochs.
    'lr': 0.0002,  # Learning rate for optimizers
    'beta1': 0.5,  # Beta1 hyperparam for Adam optimizer
    'save_epoch': 2}  # Save step.

dcgan_generator = Generator(params)
dcgan_generator.load_state_dict(models['generator'])


@app.route("/")
def base(): return "vim is dumb"


@app.get("/stylegan2")
def get_video_style_gan2():
    try:
        os.remove("video.mov")
    except Exception:
        pass
    
    num_steps = int(request.args.get("numSteps"))

    image_noises = [torch.randn(1, 512) for i in range(2)]
    mix_in = (image_noises[1] - image_noises[0]) / num_steps

    IMAGES = [stylegan2_generator(image_noises[0], None, force_fp32=True)[
        0].permute(1, 2, 0).detach().numpy()]

    for iter in range(num_steps):
        new_noise = image_noises[0] + (iter + 1) * mix_in
        image = stylegan2_generator(new_noise, None, force_fp32=True)[0].detach()
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))
        IMAGES.append(image)

    imageio.mimsave("video.mov", IMAGES)

    try:
        return send_file("video.mov", as_attachment=False)
    except FileNotFoundError:
        return "not found"


@app.get("/stylegan")
def get_video_style_gan():
    num_steps = int(request.args.get("numSteps"))
    print("this is num steps: " , num_steps)
    try:
        os.remove("video.mov")
    except Exception:
        pass

    image_noises = [torch.randn(1, 512) for i in range(2)]
    mix_in = (image_noises[1] - image_noises[0]) / num_steps

    IMAGES = [stylegan_generator(image_noises[0])[
        0].permute(1, 2, 0).detach().numpy()]

    for iter in range(num_steps):
        new_noise = image_noises[0] + (iter + 1) * mix_in
        image = stylegan_generator(new_noise)[0].detach()
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))
        IMAGES.append(image)

    imageio.mimsave("video.mov", IMAGES)

    try:
        return send_file("video.mov", as_attachment=False)
    except FileNotFoundError:
        return "not found"


@app.get("/dcgan")
def get_video_dcgan():
    try:
        os.remove("video.mov")
    except Exception:
        pass

    num_steps = int(request.args.get("numSteps"))

    image_noises = [torch.randn(1, 100, 1, 1) for i in range(2)]
    print(image_noises)
    mix_in = (image_noises[1] - image_noises[0]) / num_steps

    print(dcgan_generator(image_noises[0]).shape)
    IMAGES = [dcgan_generator(image_noises[0])[
        0].permute(1, 2, 0).detach().numpy()]
    for iter in range(num_steps):
        new_noise = image_noises[0] + (iter + 1) * mix_in
        print(new_noise.shape)
        image = dcgan_generator(new_noise)[0].detach()
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))
        IMAGES.append(image)

    imageio.mimsave("video.mov", IMAGES)

    try:
        return send_file("video.mov", as_attachment=False)
    except FileNotFoundError:
        return "not found"
    
if (__name__ == "__main__"):
    app.run()
