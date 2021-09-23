# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:10:07 2021

@author: Alex
"""

import sys
import time
from tensorflow import keras
import os
import pandas as pd
from random_word import RandomWords

from keras.callbacks import TensorBoard
from . import models


def build_data_generator(val_split):
    """
    
    Initializes an image generator that splits the data to train/val 
    
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255, validation_split=val_split)
    return datagen




def read_from_directory(datagen, src_dir, input_width,
                        input_height, batch_size, subset, shuffle, col_mode):
    """
    Accepts a keras generator, reads from directory and performs
        image augmentation. Populates the generator with processed images
        
        Parameters:
            datagen (ImageDataGenerator): keras data generator to be populated with input images
            scr_dir (str): location of input images
            input_width (int): desired resolution to train the network on
            input_height (int): desired resolution to train the network on
            batch_size (int): number of image per batch
            subset (str): training or validation subset
            shuffle (bool): shuffle image order
            col_mode (str): single_layer or multi_layer
            
        Returns:
            data_generator (ImageDataGenerator): populated image generator
        
        
    """
    
    assert subset in ['training', 'validation'], "must specify read directory as either\
    \'train\' or \'test\' "
    
    assert col_mode in ['single_layer', 'multi_layer'], "col_mode must be\
        either single_layer or multi_layer"
        
    c = 'rgb'
    if(col_mode == 'single_layer'):
        c = 'grayscale'
    

    data_generator = datagen.flow_from_directory(
        directory=src_dir,
        target_size=(input_height, input_width),
        batch_size=batch_size,
        class_mode="input",
        color_mode=c,
        shuffle=shuffle,
        subset=subset,
        seed=42)

    return data_generator



def train_model(src_dir,
                input_width,
                input_height,
                input_depth,
                learning_rate,
                epochs,
                batch_size,
                val_split,
                col_mode):
    """
    

    Parameters
    ----------
    src_dir : str
        directory of input image tiles
    input_width : int
        pixel width of input tiles
    input_height : int
        pixel height of input tiles
    input_depth : int
        number of layers in input images
    learning_rate : float
        starting learning rate for autoencoder
    epochs : int
        number of training epochs
    batch_size : int
        batch size for training
        models script
    val_split : TYPE
        DESCRIPTION.
    col_mode : TYPE
        DESCRIPTION.

    Returns
    -------
    autoencoder : TYPE
        DESCRIPTION.
    encoder : TYPE
        DESCRIPTION.
    validation_generator : TYPE
        DESCRIPTION.

    """
    
    
    available_models = models.model_choices()
    available_models = list(available_models.keys())
    model_choice = input('please type in a choice of model from the \
                         following:' + str(available_models))
                         
    assert model_choice in available_models, 'invalid model choice selected'
    
    data_generator = build_data_generator(val_split=val_split)
    train_generator = read_from_directory(data_generator, src_dir,
                                          input_width, 
                                          input_height, 
                                          batch_size,
                                          subset='training', shuffle=True, 
                                          col_mode=col_mode)
    validation_generator = read_from_directory(data_generator, src_dir,
                                               input_width,
                                               input_height,
                                               batch_size,
                                               subset='validation', shuffle=False, 
                                          col_mode=col_mode)

    #mlflow.log_param('train_samples', len(train_generator))
    #mlflow.log_param('test_samples', len(validation_generator))

    #for tag in tags:
    #    mlflow.set_tag('note', tag)
    #if model_choice:
    #    mlflow.set_tag('model', model_choice)

    # TODO pass loss_func / optimizer options/params into this interface
    autoencoder, encoder = models.build_model(x_size=input_width,
                                       y_size=input_height,
                                       z_size = input_depth,
                                       learning_rate=learning_rate,
                                       model_choice=model_choice)

    # train autoencoder
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    # STEP_SIZE_TEST = validation_generator.n // validation_generator.batch_size # noqa E501
    
    autoencoder._get_distribution_strategy = lambda: None
    
    autoencoder.fit_generator(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=validation_generator,
                        validation_steps=10,
                        epochs=epochs,
                        callbacks=[TensorBoard(log_dir='tensorboard_logs',profile_batch=0)])  # noqa E501
    
    
    r=RandomWords()
    name = r.get_random_word() + '_' + r.get_random_word()
    autoencoder.save("./models/" + name + "-autoencoder.h5")
    encoder.save("./models/" + name + "-encoder.h5")
    
    if os.path.isfile('./models/models.csv'):
       df = pd.read_csv('./models/models.csv')
       
    else:
        df = pd.DataFrame(columns = ['name', 'lr', 'tile_width', 'tile_height', 
                                     'tile_depth', 'epochs', 'batch_size',
                                     'model_type'])
        
    row = pd.DataFrame({'name': [name],
                        'lr':[learning_rate], 
                        'tile_width': [input_width], 
                        'tile_height': [input_height],
                        'tile_depth': [input_depth], 
                        'epochs': [epochs],
                        'batch_size': [batch_size],
                        'model_type': [model_choice]})
    
    df = df.append(row, ignore_index=True)
    
    df.to_csv('./models/models.csv')
    
    # TODO consider separating concerns more here rather than return list
    return autoencoder, encoder, validation_generator