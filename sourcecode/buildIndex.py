##
## Script Name: buildIndex.py
## Description: Script to load embeddings and build FAISS Indexes for Nearest Neighbour functionality
##

import pickle
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import faiss
import yaml

model_config_yaml = open('/kaggle/input/config/modelBuild.yaml', 'r')
config = yaml.load(model_config_yaml, Loader=yaml.FullLoader)
model_config_yaml.close()

#Function to get pickle file path and embeddings
def load_embeddings():
    try:
        pickle_file_path = config['feature_filename_dict']
        feature_list = pickle.load(open(pickle_file_path,'rb'))
        embeddings = np.array(list(feature_list.values()), dtype=np.float32)
        filename_df = pd.DataFrame(list(feature_list.keys()), columns=['image'])
        filename_df = filename_df.reset_index(drop=True)
        return embeddings, filename_df
    except Exception as e:
        raise Exception('Failure in get_embeddings {}'.format(str(e)))

#Function to build and save FAISS indexes developed using image embeddings
def build_faiss_index(embeddings):
    try:
        dimensions = embeddings.shape[1] #2048
        index_flatIP = faiss.IndexFlatIP(dimensions) #FlatIP indexing - Exact Search for Inner Product (Similar to Cosine Similarity)
        faiss.normalize_L2(embeddings) #Normalizing embeddings
        index_flatIP.add(embeddings) #Adding normalized embeddings to FAISS Index
        
        # Write FAISS Index
        faiss.write_index(index_flatIP, config['index_path'])

    except Exception as e:
        raise Exception('Failure in build_faiss_index {}'.format(str(e)))

#Function to build and save Annoy Tree developed using image embeddings
def build_annoy_index(embeddings):
    try:
        annoy_index = AnnoyIndex(2048,'angular')
        for i in range(embeddings.shape[0]):
            annoy_index.add_item(i, embeddings[i])

        annoy_index.build(50) # 50 trees
        annoy_index.save(config['annoy_index_path'])
        
    except Exception as e:
        raise Exception('Failure in build_annoy_index {}'.format(str(e)))

#Function to get FAISS index and ANNOY tree developed using image embeddings
def get_index():
    try:
        faiss_index_file_path = config['index_path']
        annoy_index_file_path = config['annoy_index_path']
        
        faiss_index = faiss.read_index(faiss_index_file_path)
        annoy_index = AnnoyIndex(2048, 'angular')
        annoy_index.load(annoy_index_file_path)

        return faiss_index, annoy_index
    
    except Exception as e:
        raise Exception('Failure in get_index {}'.format(str(e)))