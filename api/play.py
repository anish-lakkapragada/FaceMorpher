import os 
import random, time 
from multiprocessing import Process
from threading import Thread
import shutil
import boto3

# set up S3 

from keys import SAK, AK

access_key_id = AK
secret_access_key = SAK


s3 = boto3.resource("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

copy_source = {
    'Bucket': 'face-morpher-videos',
    'Key': 'stylegan/100.mp4'
}

#s3.meta.client.copy(copy_source, 'serve-morpher', )

# get all files in a directory

face_morpher_video_bucket = s3.Bucket('face-morpher-videos')

# get random file name 
random_file_name = random.choice([obj.key for obj in face_morpher_video_bucket.objects.filter(Prefix="stylegan")])

copy_source = {
    'Bucket': 'face-morpher-videos',
    'Key': random_file_name
}

    # puut this in the serve-morpher directory 
s3.meta.client.copy(copy_source, 'serve-morpher', random_file_name)

# getting the s3 url
bucket_name = "serve-morpher"
key = "100.mp4"

s3 = boto3.resource('s3', aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key)

bucket = s3.Bucket(bucket_name)
location = boto3.client('s3', aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key).get_bucket_location(Bucket=bucket_name)['LocationConstraint']
url = "https://s3-%s.amazonaws.com/%s/%s" % (location, bucket_name, key)
print(url)
