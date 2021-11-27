from flask import Flask, render_template, send_file, abort, send_from_directory
import imageio
import torch, numpy as np
from model import Generator
import os

app = Flask(__name__)

# load the model 

models = torch.load("dcgan_model.pth", map_location=torch.device('cpu') )
params = {
    "bsize" : 128,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 10,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2}# Save step.

generator = Generator(params) 
generator.load_state_dict(models['generator'])

# got the model loaded!
# svelte 

@app.route("/")
def base(): return "vim is dumb" 

@app.route("/getVideo/<int:num_steps>/<int:random>")
def get_video(num_steps, random): 
    try: 
        os.remove("video.mov")
    except Exception: 
        pass 
    

    image_noises = [torch.randn(1, 100, 1, 1) for i in range(2)]
    print(image_noises)
    mix_in = (image_noises[1] - image_noises[0]) / num_steps 

    print(generator(image_noises[0]).shape)
    IMAGES = [generator(image_noises[0])[0].permute(1, 2,0).detach().numpy()] 
    for iter in range(num_steps): 
        new_noise = image_noises[0] + (iter + 1) * mix_in 
        print(new_noise.shape)
        image = generator(new_noise)[0].detach() 
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))
        IMAGES.append(image)
        
    imageio.mimsave("video.mov", IMAGES)

    try:
        return send_file("video.mov", as_attachment=False)
    except FileNotFoundError:
        return "not found"
    

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    print("HAD to SEND THIS")
    return r

if __name__ == '__main__':
    app.run("0.0.0.0")
