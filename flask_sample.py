# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:16:45 2021

@author: Gayathri
"""

from flask import Flask
from flask import jsonify
from flask import request
import json
import io
import base64
from PIL import Image
import re
import sys
sys.path.insert(0,'/docker_data/sourcecode')
#from trainMultilabelModel import *
from featureExtract import *
from recommendations import *
from buildIndex import *
sys.path.insert(0,'/docker_data/models')
sys.path.insert(0,'/docker_data/indexes')
sys.path.insert(0,'/docker_data/images')
sys.path.insert(0,'/docker_data/config')


def byte_array_to_img(byte_array):
    byte_array = base64.b64decode(byte_array)
    dataBytesIO = io.BytesIO(byte_array)
    imagefile = Image.open(dataBytesIO)
    imagefile = imagefile.save("./input.jpg")
    
def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    imgByteArr = base64.b64encode(imgByteArr)
    return imgByteArr

def recommend_images():
    query = get_query_image_embedding('./input.jpg')
    faiss_index, annoy_index = get_index()
    recommended_indexes = get_recommendations(faiss_index, annoy_index, query)

def send_recommendations(recommended_indexes):
    try:
        final_image_without_mirror=[]
        final_image=[]
        for i in recommended_indexes:
            image_name = filename_df.iloc[i].image
            if '_mirror' in image_name:
                final_image_without_mirror.append(''.join(image_name.split("_mirror")))
            else:
                final_image_without_mirror.append(image_name)

        for name in final_image_without_mirror:
            if name not in final_image:
                final_image.append(name)
        with open('image_bytearray_dict.pickle', 'rb') as handle:
            image_bytearray = pickle.load(handle)
        recommendations_dict = {}
        for idx,img in enumerate(final_image[1:7], start=1):
            byte_array = image_bytearray.get(img)            
            recommendations_dict['rec_'+str(idx)]= {'recommendation_img_name_'+str(idx):img,
                                   'recommendation_img_byte_array_'+str(idx):str(byte_array)}
            #recommendations_dict.update(recommendations_dict_+str(idx))
        data['recommendations']=recommendations_dict
        return data
        #figures = {str(name): load_image(name) for name in final_image[1:7]}
        #print(figures)
    except Exception as e:
        raise Exception('Failure in send_recommendations {}'.format(str(e)))

app = Flask(__name__)


@app.route('/processjson',methods=['POST'])
def processjson():
    input_json_bytearray = request.get_json()
    email_id = input_json_bytearray['emailId']
    byte_array=input_json_bytearray['byteArrayOutput']
    imageBytes = re.split(',',byte_array)[1]
    #Call bytearray to img conversion function
    byte_array_to_img(imageBytes)
    #Pass image to get recommendations function
    recommended_indexes = recommend_images()
    #Call send recommendation function here instead of reading the recommendations_data.json sample
    recommendations = send_recommendations(recommended_indexes)
    #output = {}
    #with open('recommendations_data.json') as f:
    #    data = json.load(f)
    #    recommendations = data['recommendations']
    output['input_json_bytearray'] = input_json_bytearray
    output['email'] = email_id
    output['recommendations'] = recommendations
    return jsonify(output)

if __name__ == '__main__':
    app.run(host = '0.0.0.0')