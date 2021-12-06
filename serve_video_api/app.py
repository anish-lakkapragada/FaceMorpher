from flask import Flask, send_file, request, after_this_request, current_app
from flask_cors import CORS, cross_origin
from pathlib import Path
import os 
import random, time 
#from multiprocessing import Process
#from threading import Thread
#import shutil 
import boto3

# set up S3 

from keys import SAK, AK

access_key_id = AK
secret_access_key = SAK

s3 = boto3.resource("s3", aws_access_key_id=access_key_id, 
                        aws_secret_access_key=secret_access_key)

face_morpher_video_bucket = s3.Bucket('face-morpher-videos')
serve_video_bucket = s3.Bucket("serve-morpher")

#location = boto3.client('s3', aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key).get_bucket_location(Bucket="serve-morpher")['LocationConstraint']
location="us-west-1"
print("THIS IS LOCATION : " + location)

svelte_app = str(Path(__file__).parent / "../public")

app = Flask(__name__, static_url_path="", static_folder=svelte_app)
api_v1_cors_config = {
    "origins": ["*"], 
    "methods": ["OPTIONS", 'GET', 'POST'], 
    "allow_headers": ["Authorization", "Content-Type"]
}

CORS(app, resources={r"/*" : api_v1_cors_config})


import os 
def video_file(model_type): 
    """give back a random file from the s3 bucket 
    and transfer it to serve/ bucket and return file name"""
    
    # get random file name 
    random_file_name = random.choice([obj.key for obj in face_morpher_video_bucket.objects.filter(Prefix=model_type)])

    copy_source = {
        'Bucket': 'face-morpher-videos',
        'Key': random_file_name
    }

    # puut this in the serve-morpher directory 
    s3.meta.client.copy(copy_source, 'serve-morpher', random_file_name)
    
    # delete this from the face-morpher-videos bucket 
    s3.Object('face-morpher-videos', random_file_name).delete()

    return random_file_name
    
def create_video(model): 
    os.system(f"python3.8 create_video.py {model}")

@app.route("/")
@cross_origin(**api_v1_cors_config)
def base():
    with open(Path(svelte_app) / "index.html", "r") as fin:
        return fin.read()

@app.get("/dcgan")
@cross_origin(**api_v1_cors_config)
def dcgan_video(): 
    video_file_name = video_file("dcgan")
    return "https://s3-%s.amazonaws.com/%s/%s" % (location, "serve-morpher", video_file_name)

@app.get("/stylegan2")
@cross_origin(**api_v1_cors_config)
def stylegan2_video(): 
    video_file_name = video_file("stylegan2")
    return "https://s3-%s.amazonaws.com/%s/%s" % (location, "serve-morpher", video_file_name)

@app.get("/stylegan")
@cross_origin(**api_v1_cors_config)
def stylegan_video(): 
    video_file_name = video_file("stylegan")
    
    # got it from here, just send the file 
    return "https://s3-%s.amazonaws.com/%s/%s" % (location, "serve-morpher", video_file_name)
    
if __name__ == "__main__": 
    app.run() 