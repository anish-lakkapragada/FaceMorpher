"""
Script to generate a video of a certain type. 
"""

import torch
import math, random
import imageio 
import os
from load_models import load_dcgan, load_stylegan, load_stylegan2

max_dcgan_num = max([int(file_name[:file_name.index(".")]) for file_name in os.listdir("videos/dcgan")])
max_stylegan_num = max([int(file_name[:file_name.index(".")]) for file_name in os.listdir("videos/stylegan")])
max_stylegan2_num = max([int(file_name[:file_name.index(".")]) for file_name in os.listdir("videos/stylegan2")])

VIDEO_NUM = max([max_dcgan_num, max_stylegan_num, max_stylegan2_num])

def make_video(type_model, generator, latent_feed_function, latent_size, num_steps=20): 
    global VIDEO_NUM
    if type_model == "dcgan": 
        image_noises = [torch.randn(1, 100, 1, 1) for i in range(2)]
    else: 
        image_noises = [torch.randn(1, 512) for i in range(2)]
    
    mix_in = (image_noises[1] - image_noises[0]) / num_steps
    IMAGES = [latent_feed_function(generator, image_noises[0])] # takes care of all detaching, etc. 

    for i in range(num_steps): 
        new_noise = image_noises[0] + (i + 1) * mix_in
        image = latent_feed_function(generator, new_noise)
        IMAGES.append(image)
    
    imageio.mimsave("videos/" + type_model + f"/{str(VIDEO_NUM)}.mov", IMAGES)

def style_gan_feed_function(generator, latent): 
    return generator(latent).squeeze(0).permute(1, 2, 0).detach().numpy()

def style_gan2_feed_function(generator, latent): 
    return generator(latent, None, force_fp32 = True).squeeze(0).permute(1, 2, 0).detach().numpy()

def dcgan_feed_function(generator, latent): 
    return generator(latent).squeeze(0).permute(1, 2, 0).detach().numpy()

video_types = ['dcgan', 'stylegan', 'stylegan2']
new_video_index = 0
latent_feed_functions = {"dcgan": dcgan_feed_function, "stylegan": style_gan_feed_function, "stylegan2": style_gan2_feed_function}
latent_sizes = {"dcgan": 100, "stylegan": 512, "stylegan2": 512}
generators = {"dcgan": load_dcgan(), "stylegan": load_stylegan(), "stylegan2": load_stylegan2()}

# read in the arguments
import sys 
model_type = sys.argv[1]
make_video(model_type, generators[model_type], latent_feed_functions[model_type], latent_sizes[model_type])

# this creates the video.