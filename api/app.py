from flask import Flask, send_file, request, after_this_request, current_app
from flask_cors import CORS, cross_origin
import os 
import random, time 
from multiprocessing import Process
from threading import Thread
import shutil 

"""
When a request is received, pull a random video from the
videos directory, and return it back. Then run create_video.py again to create a video 
for whichever type is gone now. 

If there are currently no videos in that pool, generate one as well!
"""

app = Flask(__name__)
api_v1_cors_config = {
    "origins": ["*"], 
    "methods": ["OPTIONS", 'GET', 'POST'], 
    "allow_headers": ["Authorization", "Content-Type"]
}

CORS(app, resources={r"/*" : api_v1_cors_config})


import os 
def video_file(model_type): 
    model_files = os.listdir("videos/" + model_type)
    if len(model_files) == 0: 
        # create a model file 
        os.system(f"python3.8 create_video.py {model_type}")
    
    model_video_file = random.choice(os.listdir(f"videos/{model_type}"))
    
    return f"videos/{model_type}/{model_video_file}"


def create_video(model): 
    os.system(f"python3.8 create_video.py {model}")

@app.route("/")
@cross_origin(**api_v1_cors_config)
def base(): return "vim is fun"

@app.get("/dcgan")
@cross_origin(**api_v1_cors_config)
def dcgan_video(): 
    video_file_name = video_file("dcgan")
    new_path = "../public/serve/dcgan"
    # delete all videos from here 
    shutil.rmtree(new_path)
    os.mkdir(new_path) # clear all 

    shutil.move(video_file_name, new_path)
    return os.listdir(new_path)[0]

@app.get("/stylegan2")
@cross_origin(**api_v1_cors_config)
def stylegan2_video(): 
    video_file_name = video_file("stylegan2")
    new_path = "../public/serve/stylegan2"
    # delete all videos from here 
    shutil.rmtree(new_path)
    os.mkdir(new_path) # clear all 

    shutil.move(video_file_name, new_path)
    return os.listdir(new_path)[0]

@app.get("/stylegan")
@cross_origin(**api_v1_cors_config)
def stylegan_video(): 
    video_file_name = video_file("stylegan")
    
    new_path = "../public/serve/stylegan"
    # delete all videos from here 
    shutil.rmtree(new_path)
    os.mkdir(new_path) # clear all 

    shutil.move(video_file_name, new_path)
    return os.listdir(new_path)[0]

@app.get("/serve/<model_type>/<video_name>")
@cross_origin(**api_v1_cors_config)
def give_video(model_type, video_name): 
    return send_file(f"../public/serve/{model_type}/{video_name}", as_attachment=True)
    
if __name__ == "__main__": 
    app.run() 