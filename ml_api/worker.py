"""
Script to continously generate videos. Saves them in the appropriate videos/ directory. 
"""

import torch
import math, random
import imageio 
import os, boto3
from load_models import load_dcgan, load_stylegan, load_stylegan2
from boto3.s3.transfer import S3Transfer
import boto3

from keys import SAK, AK

NUM_MAX_VIDEOS = 200
VIDEO_NUM = 60

access_key_id = AK
secret_access_key = SAK

s3 = boto3.client("s3", aws_access_key_id=access_key_id, 
                        aws_secret_access_key=secret_access_key)

def create_video(type_model, generator, latent_feed_function, latent_size, num_steps=20): 
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
    
    FILE_PATH ="videos/" + type_model + f"/{str(VIDEO_NUM)}.mp4"
    imageio.mimsave(FILE_PATH, IMAGES)

    # save to bucket 
    s3.upload_file(FILE_PATH, "face-morpher-videos", f"{type_model}/{str(VIDEO_NUM)}.mp4")

    VIDEO_NUM += 1

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

while True: 
    type_model = video_types[new_video_index]
    print(f"working on this model: {type_model}")
    latent_feed_function = latent_feed_functions[type_model]

    if len(os.listdir("videos/" + type_model)) > NUM_MAX_VIDEOS: 
        new_video_index += 1
        if new_video_index == len(video_types): new_video_index = 0 
        continue # we ain't doing this one either 

    # create the video 
    create_video(type_model, generators[type_model], latent_feed_function, latent_sizes[type_model])
      
    new_video_index += 1 
    if new_video_index == len(video_types): new_video_index = 0