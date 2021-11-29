from flask import Flask, send_file, request, after_this_request, send_from_directory
import os 
import random, time 
from multiprocessing import Process

"""
When a request is received, pull a random video from the
videos directory, and return it back. Then run create_video.py again to create a video 
for whichever type is gone now. 

If there are currently no videos in that pool, generate one as well!
"""

app = Flask(__name__)

def video_file(model_type): 
    model_files = os.listdir("videos/" + model_type)
    if len(model_files) == 0: 
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
    def remove_file(response):
        try:
            os.remove(video_file_name)
            #video_file_name.close()
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response
    return send_file(video_file_name, as_attachment=True)

@app.get("/stylegan2")
def stylegan2_video(): 
    video_file_name = video_file("stylegan2")
    @after_this_request
    def remove_file(response):
        try:
            os.remove(video_file_name)
            #video_file_name.close()
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response
    return send_file(video_file_name, as_attachment = True)

@app.get("/stylegan")
def stylegan_video(): 
    video_file_name = video_file("stylegan")
    print("got it here : ", video_file_name)
    @after_this_request
    def remove_file(response):
        try:
            os.remove(video_file_name)
            #video_file_name.close()
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response
    return send_file(video_file_name, as_attachment = True)

if __name__ == "__main__": 
    app.run() 