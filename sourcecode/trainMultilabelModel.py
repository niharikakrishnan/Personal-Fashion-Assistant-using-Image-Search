##
## Script Name: preProcess.py
## Description: Script to train and build models
##

import os
import numpy as np
import pandas as pd
import itertools
import pickle
import keras
import datetime
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.layers import Input, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing import image
from keras.models import Model
import multiprocessing
import ast
import yaml

# config = pickle.load(open('config.pkl','rb'))
model_config_yaml = open('/kaggle/input/config/modelBuild.yaml', 'r')
config = yaml.load(model_config_yaml, Loader=yaml.FullLoader)
model_config_yaml.close()

input_path = config['input_path']
imgs_path = config['imgs_path']
# Read Tuple data
input_shape = ast.literal_eval(config['input_shape'])
target_size = ast.literal_eval(config['target_size'])

# Function to create Multi-label datasets
def multi_label_dataset(dataset_csv):
    try:
        styles_csv = pd.read_csv(input_path+dataset_csv,error_bad_lines=False)
        # Filter for Apparel and Shoes
        styles_csv = styles_csv[(styles_csv['masterCategory']=='Apparel')|(styles_csv['masterCategory']=='Footwear')]
        # Create labels for multi-label classification
        styles_csv['Filenames'] = styles_csv['img_nm']  
        #styles_csv['Filenames'] = styles_csv['id'].apply(lambda x: str(x)+".jpg")
        columns = ['gender','masterCategory','subCategory','articleType','usage']
        styles_csv['labels'] = styles_csv[columns].values.tolist()
        final_df = styles_csv[['Filenames','labels']]
        # Remove nans from created labels
        final_df['labels'] = final_df['labels'].apply(lambda x: [i for i in x if i == i])
        final_df.to_excel('final_df.xlsx')
        return final_df

    except Exception as e:
        raise Exception('Failure in multilabel_predict {}'.format(str(e)))

# Function to extract class list and # of classes in the dataset
def get_class_list(df):
    try:
        # Get all possibe labels and count of total labels to feed to image data generator
        list_of_labels = df.labels.tolist()
        classes = itertools.chain.from_iterable(list_of_labels)
        classes = list(set(classes))
        num_classes = len(classes)
        return classes, num_classes

    except Exception as e:
        raise Exception('Failure in get_class_list {}'.format(str(e)))

# Function to create train, validation, and test image data generators
def img_data_generators(final_df, shuffle_fn):
    try:
        classes, num_classes = get_class_list(final_df)
        # Use keras image data generators to create train, test and validate sets
        datagen = ImageDataGenerator(rescale=1./255.)
        test_datagen = ImageDataGenerator(rescale=1./255.)
        # Get indices for 80-10-10 ratio for train-valid-test split
        total_dataset_size = len(final_df)
        train_data_idx = round(0.8 * total_dataset_size)
        val_data_idx = train_data_idx + round(0.1 * total_dataset_size)
        # Shuffle the dataframe
        if shuffle_fn == 'Y':
            final_df = final_df.sample(frac=1).reset_index(drop=True)
        
        # Create train data generator
        train_generator = datagen.flow_from_dataframe(dataframe=final_df[:train_data_idx],
                                                    directory=imgs_path,
                                                    x_col="Filenames",
                                                    y_col="labels",
                                                    batch_size=config['train_batch_size'],
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    classes=classes,
                                                    target_size=target_size)
        
        # Create validation image generator
        valid_generator = test_datagen.flow_from_dataframe(dataframe=final_df[train_data_idx:val_data_idx],
                                                    directory=imgs_path,
                                                    x_col="Filenames",
                                                    y_col="labels",
                                                    batch_size=config['valid_batch_size'],
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    classes=classes,
                                                    target_size=target_size)
        
        # Create test image generator
        test_generator = test_datagen.flow_from_dataframe(dataframe=final_df[val_data_idx:total_dataset_size],
                                                    directory=imgs_path,
                                                    x_col="Filenames",
                                                    batch_size=config['test_batch_size'],
                                                    seed=42,
                                                    shuffle=False,
                                                    class_mode=None,
                                                    target_size=target_size)
        
        return train_generator,valid_generator,test_generator

    except Exception as e:
        raise Exception('Failure in img_data_generators {}'.format(str(e)))

# Function to build the model
def build_model(model,unfreeze,df):
    try:
        classes,num_classes = get_class_list(df)

        # Load Xception or Efficient Net as the base model
        if model == 'xception':
            base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape = input_shape)
            base_model.trainable = False    # if compute capabilities available, can even try setting base model weight trainable to True
            inputs = keras.Input(shape=input_shape)
        elif model=='efficient net':
            base_model = tf.keras.applications.EfficientNetB5(weights='imagenet', include_top=False,input_shape = input_shape)
            base_model.trainable = False    # if compute capabilities available, can even try setting base model weight trainable to True
            inputs = keras.Input(shape=input_shape)
        
        #x = data_augmentation(inputs)  # Apply random data augmentation
        x = inputs
        # Pre-trained Xception weights requires that input be normalized
        # from (0, 255) to a range (-1., +1.), the normalization layer
        # does the following, outputs = (inputs - mean) / sqrt(var)
        norm_layer = keras.layers.experimental.preprocessing.Normalization()
        mean = np.array([127.5] * 3)
        var = mean ** 2
        # Scale inputs to [-1, +1]
        x = norm_layer(x)
        norm_layer.set_weights([mean, var])
        x = base_model(x, training=False)
        x = base_model.output
        x = GlobalMaxPooling2D(name='max_pool')(x)
        x = Dropout(0.4)(x)
        x = Dense(num_classes, activation='sigmoid',name = 'dense_layer')(x)
        model = Model(inputs=base_model.input, outputs=x)
        if unfreeze=='Y':
            for layer in model.layers[:config['unfreeze_layer_idx']]:
                layer.trainable = False
            for layer in model.layers[config['unfreeze_layer_idx']:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
        model.summary()
        return model

    except Exception as e:
        raise Exception('Failure in build_model {}'.format(str(e)))
 
# Function to Train or Loading a trained model
def train_or_load_model(final_df,training):
    try:
        # Get train, valid and test generators
        train_generator,valid_generator,test_generator = img_data_generators(final_df, config['dataset_shuffle'])
        model = build_model(config['model_name'],config['unfreeze_layers'],final_df)
        model.compile(optimizers.Adam(lr=config['learning_rate']),loss = keras.losses.BinaryCrossentropy(from_logits=True),metrics=["accuracy"])
        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
        STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

        if training=='Y':
            filepath = config['model_save_filepath']
            #Hi Bharath! I haven't added the tensorboard callback here as I'm not 
            #sure if that will be supported in the MFDM workbench. Add it if you can
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            early_stopping = EarlyStopping(monitor='loss', patience=config['early_stopping_patience'])
            model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=valid_generator,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=config['epochs'], callbacks=[checkpoint,early_stopping],verbose = 1
        )
            model.save(config['final_model_save_path'])
            return model,train_generator,test_generator,STEP_SIZE_TEST
        else:
            # model = tf.keras.models.load_model(config['final_model_save_path'], custom_objects=None, compile=True, options=None)
            model = load_saved_model(config['final_model_save_path'])
            return model, train_generator, test_generator, STEP_SIZE_TEST
    
    except Exception as e:
        raise Exception('Failure in train_or_load_model {}'.format(str(e)))

def load_saved_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)
        return model
    
    except Exception as e:
        raise Exception('Failure in load_saved_model {}'.format(str(e)))