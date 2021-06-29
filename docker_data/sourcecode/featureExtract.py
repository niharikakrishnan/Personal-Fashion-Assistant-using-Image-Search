##
## Script Name: preProcess.py
## Description: Script to extract embeddings for images
##

import keras
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import ast
import yaml

# config = pickle.load(open('config.pkl','rb'))
model_config_yaml = open('/docker_data/config/modelBuild.yaml', 'r')
config = yaml.load(model_config_yaml, Loader=yaml.FullLoader)
model_config_yaml.close()

# Read Tuple data
target_size = ast.literal_eval(config['target_size'])

# Function to predict multi-label classifiation
def multilabel_predict(model, train_generator, test_generator, step_size_test):
    try:
        test_generator.reset()
        pred = model.predict_generator(test_generator,steps=step_size_test,verbose=1)
        pred_bool = (pred >0.5)
        predictions =[]
        labels = train_generator.class_indices
        labels = dict((v,k) for k,v in labels.items())
        for row in pred_bool:
            l=[]
            for index,clas in enumerate(row):
                if clas:
                    l.append(labels[index])
            predictions.append(",".join(l))
        filenames=test_generator.filenames
        results=pd.DataFrame({"Filename":filenames,
                            "Predictions":predictions})
        results.to_csv(config['multi_label_prediction_results'],index=False)
        return results

    except Exception as e:
        raise Exception('Failure in multilabel_predict {}'.format(str(e)))

# Function to extract features from multi-label image classification model for a given image
def get_embedding(model, img_name):
    try:
        # Reshape a given image
        img = keras.preprocessing.image.load_img(img_name, target_size=target_size)
        # Convert image to Array
        x = keras.preprocessing.image.img_to_array(img)
        # Expand Dim (1, w, h)
        x = np.expand_dims(x, axis=0)
        # Pre process Input
        #x = preprocess_input(x)
        layer_name = 'max_pool'
        intermediate_layer_model = keras.Model(inputs=model.input,
                                            outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(x)
        intermediate_output = intermediate_output.numpy().reshape(-1)
        return intermediate_output

    except Exception as e:
        raise Exception('Failure in get_embedding {}'.format(str(e)))

# Function to save embeddings for all input images or load existing embdeddings
def extract_embedding(model, extract, filenames):
    try:
        if extract =='Y':
            feature_list = []
            feature_filename_dict = {}
            #it may be a good idea to add a try except inside this for loop and also pop 
            #images from the filename list in except block if embedding extraction fails
            #for any particular image
            for i in filenames:
                embedding = get_embedding(model,(config['imgs_path']+i))
                feature_list.append(embedding)
                feature_filename_dict[i] = embedding
            pickle.dump(feature_list, open(config['feature_list'], 'wb'))
            pickle.dump(filenames, open(config['filename_list'],'wb'))
            pickle.dump(feature_filename_dict,open(config['feature_filename_dict'],'wb'))
        else:
            feature_list = pickle.load(open(config['feature_list'],'rb'))
            filenames = pickle.load(open(config['filename_list'],'rb'))
            feature_filename_dict = pickle.load(open(config['feature_filename_dict'],'rb'))
        return filenames, feature_list, feature_filename_dict

    except Exception as e:
        raise Exception('Failure in extract_embedding {}'.format(str(e)))
