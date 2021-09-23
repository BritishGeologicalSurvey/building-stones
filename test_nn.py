# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:23:33 2021

@author: Alex
"""

from tensorflow import keras
from SIMILE import nn_testing

#location of trained encoder
MODEL = "./models/urbanness_logogrammatic-encoder.h5"

encoder = keras.models.load_model(".\\models\\gaolers_divergency-encoder.h5")

actuals, freqs = nn_testing.test_model("./sample_processed/blue", encoder, 256,256, 'grayscale' )

prediction_summary = nn_testing.make_prediction_summary(freqs, actuals)

nn_testing.plot_confusion_matrices(prediction_summary, actuals)
nn_testing.plot_confidence_matrix(prediction_summary, actuals)
