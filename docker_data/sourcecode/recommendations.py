##
## Script Name: recommendations.py
## Description: Script to get embeddings and recommend similar images
##

from buildIndex import get_index, load_embeddings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featureExtract import get_embedding
from trainMultilabelModel import load_saved_model
import cv2
import yaml

model_config_yaml = open('/docker_data/config/modelBuild.yaml', 'r')
config = yaml.load(model_config_yaml, Loader=yaml.FullLoader)
model_config_yaml.close()

# Load Model from disk
model = load_saved_model(config['final_model_save_path'])
embeddings, filename_df = load_embeddings()

# Function to format the recommended images grid structure
def plot_figures(figures, nrows = 1, ncols=1,figsize=(8, 8)):
    try:
        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
        for ind,title in enumerate(figures):
            axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout()
    except Exception as e:
        raise Exception('Failure in plot_figures {}'.format(str(e)))

# Function to load the image path based on the image filename
def img_path(img):
    try:
        return config['imgs_path'] + img
    except Exception as e:
        raise Exception('Failure in img_path {}'.format(str(e)))

# Function to read and resize image using OpenCV functionalities
def load_image(image):
    try:
        img = cv2.imread(img_path(image))
        w, h, _ = img.shape
        resized = cv2.resize(img, (299,299), interpolation = cv2.INTER_AREA)
        return resized
    except Exception as e:
        raise Exception('Failure in load_image {}'.format(str(e)))

# Function to get input query image embeddings
def get_query_image_embedding(query_image_path):
    try:
        image_embedding = get_embedding(model, query_image_path)
        # Reshaping embedding dimensionality
        query = np.asarray(image_embedding, dtype=np.float32)
        return query
    except Exception as e:
        raise Exception('Failure in get_query_image_embedding {}'.format(str(e)))

# Function to get distances and indexes of recommended images using FAISS & ANNOY methods
def get_recommendations(faiss_index, annoy_index, query):
    try:
        recommendations = 20
        faiss_distances, faiss_indexes = faiss_index.search(query.reshape(1,2048), recommendations)
        annoy_indexes, annoy_distances = annoy_index.get_nns_by_vector(query, recommendations, include_distances=True)
        final_index = []
        for (i,j) in zip(faiss_indexes[0], annoy_indexes):
            if i==j:
                final_index.append(i)
            else:
                final_index.append(i)
                final_index.append(j)
            
            if len(final_index)>=20:
                return final_index

    except Exception as e:
        raise Exception('Failure in get_recommendations {}'.format(str(e)))

# Function to display recommended images
def display_recommended_images(recommended_indexes):
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
        
        figures = {str(name): load_image(name) for name in final_image[1:7]}
        
        # Plot 6 recommendations in a 2X3 matrix format
        plot_figures(figures, 2, 3)
    except Exception as e:
        raise Exception('Failure in display_recommended_images {}'.format(str(e)))