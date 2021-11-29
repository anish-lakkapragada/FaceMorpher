from flask import Flask, send_file, request, after_this_request
import os 
import random, time 

"""
When a request is received, pull a random video from the
videos directory, and return it back. Then run create_video.py again to create a video 
for whichever type is gone now. 

If there are currently no videos in that pool, generate one as well!
"""

app = Flask(__name__)
DCGAN_VIDEO_NAME_REMOVED = None 
STYLEGAN_VIDEO_NAME_REMOVED = None
STYLEGAN2_VIDEO_NAME_REMOVED = None 

def video_file(model_type): 
    model_files = os.listdir("videos/" + model_type)
    if (len(model_files) == 0): 
        # create a model file 
        os.system(f"python3.8 create_video.py {model_type}")
    
    model_video_file = random.choice(os.listdir(f"videos/{model_type}"))
    
    return f"videos/{model_type}/{model_video_file}"

def clean_and_create_model(model_type, file_name): 
    os.remove(file_name) 
    os.system(f"python3.8 create_video.py {model_type}")
    print("done cleaning and creating!")
    return 

@app.route("/")
def base(): return "vim is fun"

@app.get("/dcgan")
def dcgan_video(): 
    video_file_name = video_file("dcgan")
    @after_this_request 
    def clean_and_create(response): 
        clean_and_create_model("dcgan", video_file_name)
        return response
    return send_file(video_file_name)

@app.get("/stylegan2")
def stylegan2_video(): 
    video_file_name = video_file("stylegan2")
    @after_this_request 
    def clean_and_create(response): 
        clean_and_create_model("stylegan2", video_file_name)
        return response 
    return send_file(video_file_name)

@app.get("/stylegan")
def stylegan_video(): 
    video_file_name = video_file("stylegan")
    @after_this_request 
    def clean_and_create(response): 
        clean_and_create_model("stylegan", video_file_name)
        return response
    return send_file(video_file_name)

if __name__ == "__main__": 
    app.run() 