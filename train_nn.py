# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:50:00 2021

@author: Alex
"""

import sys
# import mlflow.keras

#from keras.callbacks import TensorBoard

from SIMILE import nn_training

#mlflow.keras.autolog()

BATCH_SIZE = 2
LEARNING_RATE = 0.0001#0.0001
EPOCHS = 2
X_SIZE = 256  # 236#124
Y_SIZE = 256 # 188#92
VAL_SPLIT = 0.1
SRC_DIR = 'C:\\Users\\Alex\\Documents\\data_science\\SIMILE\\sample_processed\\stacked'
INPUT_LAYERS = 1 #layers in the input image


col_mode = 'single_layer'
if(INPUT_LAYERS == 3):
    col_mode = 'multi_layer'
    
    
    
    
if __name__ == '__main__':
    nn_training.train_model(SRC_DIR,input_width=X_SIZE,
                input_height=Y_SIZE,
                input_depth = INPUT_LAYERS,
                learning_rate=LEARNING_RATE,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                val_split = VAL_SPLIT,
                col_mode = col_mode)
    
    