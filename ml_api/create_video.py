"""
Script to generate a video of a certain type. 
"""

import torch
import math, random
import imageio 
import os
from load_models import load_dcgan, load_stylegan, load_stylegan2
from boto3.s3.transfer import S3Transfer
import boto3

from keys import SAK, AK

access_key_id = AK
secret_access_key = SAK

s3 = boto3.client("s3", aws_access_key_id=access_key_id, 
                        aws_secret_access_key=secret_access_key)

s3_resource = boto3.resource("s3", aws_access_key_id=access_key_id, 
                        aws_secret_access_key=secret_access_key) 
face_morpher_video_bucket = s3_resource.Bucket('face-morpher-videos')

max_num = 0 
for model in ['dcgan', 'stylegan', 'stylegan2']: 
    for object in face_morpher_video_bucket.objects.filter(Prefix=model): 
        try: 
            number = object.key[object.key.index("/") + 1 : object.key.index(".")]
            max_num = max([max_num, int(number)])
        except Exception as e: 
            pass 

print(max_num)

VIDEO_NUM = max_num + 1

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
    
    FILE_PATH ="videos/" + type_model + f"/{str(VIDEO_NUM)}.mp4"
    imageio.mimsave(FILE_PATH, IMAGES)

    # save to bucket 
    s3.upload_file(FILE_PATH, "face-morpher-videos", f"{type_model}/{str(VIDEO_NUM)}.mp4")

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
print("running this dank")
make_video(model_type, generators[model_type], latent_feed_functions[model_type], latent_sizes[model_type])

# this creates the video.