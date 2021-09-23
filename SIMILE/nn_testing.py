# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 08:59:31 2021

@author: Alex
"""



import os
import pandas as pd
import pickle
import re
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
from . import nn_training
from tensorflow import keras
from scipy import spatial
from matplotlib import pyplot as plt



def buildFreqTable(aList):
    """
    convert a list of predictions to a frequency table

    Parameters
    ----------
    aList : str
        list of predictions

    Returns
    -------
    freqDict : dict
        frequency table of predictions

    """

    
    freqDict = {}
    for num in aList:
        if num in freqDict:
            freqDict[num] += 1
        else:
            freqDict[num] = 1
    return freqDict


def test_model(src_dir, model, size_x, size_y, col_mode):

    
    labels = os.listdir(os.path.join(src_dir,"test","test"))
    
    
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = datagen.flow_from_directory(
        directory=os.path.join(src_dir,"test"),
        target_size=(size_y, size_x),
        color_mode=col_mode,
        batch_size=1,
        class_mode="input",
        shuffle=False,
        seed=57)
    

    encoded_imgs = model.predict_generator(generator=test_generator)
    
    actuals = []
    predictions = []
    prediction_scores = [] #number of correct votes for each prediction
    prediction_frequencies = pd.DataFrame()# each row is a freq table for each image
    
    #compress and pickle encoded images
    encoded_dictionary = dict(zip(labels,encoded_imgs))
    pickle.dump(encoded_dictionary,open('encoded_images.p','wb'))
    
    i=0
    previous_sample_name=str()
    sample_predictions = []
    for key, value in encoded_dictionary.items():
    
        cosine_similarities = list()
        labels_list = list()
        
        #get key for the building stone sample so we compare at a sample level
        sample_key = key.split('IMG')[0]
    
        
        for key1, value1 in encoded_dictionary.items():
            sample_key1 = key1.split('IMG')[0]
    
            
            #dont compare to subsets from the input image
            if(sample_key1==sample_key):
                continue
            cosine_similarities.append(spatial.distance.cosine(value,value1))
            labels_list.append(key1)
            
        image_label = labels_list[cosine_similarities.index(sorted(cosine_similarities)[0])]
        
        prediction = image_label.split('_')[0]
        prediction = re.sub(r'[^A-Za-z]', '', prediction)
        
        if(i==0 or previous_sample_name == sample_key):
            sample_predictions.append(prediction)
    
        else:
            #count most common prediction for overall input image an append to predictions list
            #image_prediction = most_common(sample_predictions) 
            #predictions.append(image_prediction)
            
            #as above but using frequency table        
            sample_freq_table = buildFreqTable(sample_predictions)
            image_prediction = max(sample_freq_table, key=sample_freq_table.get)
            prediction_scores.append(max(sample_freq_table.values()))
            sample_freq_table = pd.DataFrame({k: [v] for k, v in sample_freq_table.items()})
            prediction_frequencies = pd.concat([prediction_frequencies, sample_freq_table], axis=0, ignore_index=True)
            predictions.append(image_prediction)
            
            
            
            #append previous sample name to actuals (alpha only)
            actual=re.sub(r'[^A-Za-z]', '', previous_sample_name)
            actual = actual[:4]
            actuals.append(actual)
            
            #reset labels list to include the new label only
            sample_predictions = [prediction]
        
        #actuals.append(key)
        #predictions.append(image_label)
        
    
        i=i+1
        print(str(i) + sample_key + prediction)
        previous_sample_name = sample_key
        
    #appeand the final prediction and actual to the list
    sample_freq_table = buildFreqTable(sample_predictions)
    image_prediction = max(sample_freq_table, key=sample_freq_table.get) 
    predictions.append(image_prediction)
    prediction_scores.append(max(sample_freq_table.values()))
    #append previous sample name to actuals (alpha only)
    actual=re.sub(r'[^A-Za-z]', '', previous_sample_name)
    actual = actual[:4]
    actuals.append(actual)
        
    sample_freq_table = pd.DataFrame({k: [v] for k, v in sample_freq_table.items()})
    prediction_frequencies = pd.concat([prediction_frequencies, sample_freq_table], axis=0, ignore_index=True)
    
    return actuals, prediction_frequencies



def make_prediction_summary(prediction_frequencies, actuals):
    """
    

    Parameters
    ----------
    prediction_frequencies : pandas table
        frequency table from the test_model method
    actuals: list
        actual classes of test set

    Returns
    -------
    a table of prediction summaries

    """
    
    NUMBER_OF_SAMPLES = len(prediction_frequencies.columns)


    #HAVE TO NAN dup;licates after each iteration. should really refactor this
    prediction_summary = prediction_frequencies.copy(deep=True)
    prediction_summary['total_votes'] = prediction_frequencies.sum(axis=1)
    prediction_summary['actual'] = actuals
    
    prediction_frequencies = prediction_frequencies.fillna(0)
    
    #gets rid of duplicates. Is a pretty stupid bodge but works well
    prediction_frequencies = prediction_frequencies +\
    np.random.rand(*prediction_frequencies.shape) / 100.0
        
    
    prediction_frequencies = prediction_frequencies.T.apply(lambda x: x.nsmallest(NUMBER_OF_SAMPLES)).transpose()
    prediction_summary['predicted'] = prediction_frequencies.T.apply(lambda x: x.nlargest(1).idxmax())
    prediction_summary['confidence'] = 100*prediction_frequencies.T.apply(lambda x: x.nlargest(1)).min() /\
        prediction_summary['total_votes']
    
    
    prediction_frequencies = prediction_frequencies.T.apply(lambda x: x.nsmallest(NUMBER_OF_SAMPLES-1)).transpose()
    prediction_summary['prediction2'] = prediction_frequencies.T.apply(lambda x: x.nlargest(2).idxmax())
    prediction_summary['confidence2'] = 100*prediction_frequencies.T.apply(lambda x: x.nlargest(1)).min()   /\
        prediction_summary['total_votes']
    
    
    prediction_frequencies = prediction_frequencies.T.apply(lambda x: x.nsmallest(NUMBER_OF_SAMPLES-2)).transpose()
    prediction_summary['prediction3'] = prediction_frequencies.T.apply(lambda x: x.nlargest(3).idxmax())
    prediction_summary['confidence3'] = 100*prediction_frequencies.T.apply(lambda x: x.nlargest(1)).min()  /\
        prediction_summary['total_votes']
    
    
    prediction_frequencies = prediction_frequencies.T.apply(lambda x: x.nsmallest(NUMBER_OF_SAMPLES-3)).transpose()
    prediction_summary['prediction4'] = prediction_frequencies.T.apply(lambda x: x.nlargest(4).idxmax())
    prediction_summary['confidence4'] = 100*prediction_frequencies.T.apply(lambda x: x.nlargest(1)).min()   /\
        prediction_summary['total_votes']
    
    
    prediction_frequencies = prediction_frequencies.T.apply(lambda x: x.nsmallest(NUMBER_OF_SAMPLES-4)).transpose()
    prediction_summary['prediction5'] = prediction_frequencies.T.apply(lambda x: x.nlargest(5).idxmax())
    prediction_summary['confidence5'] = 100*prediction_frequencies.T.apply(lambda x: x.nlargest(1)).min()   /\
        prediction_summary['total_votes']
    
    
    prediction_summary = prediction_summary.round()
    
    return prediction_summary


def plot_confusion_matrices(prediction_summary, actuals):
    """
    plots confusion matrices for top predictions

    Parameters
    ----------
    prediction_summary : pandas dataframe
        prediction summary df from the make_prediction_summary method
    actuals: list
        actual classes

    Returns
    -------
    None.

    """
    #confusion matrices
    colnames = sorted(list(set(actuals)))
    
    cm = confusion_matrix(prediction_summary['actual'], prediction_summary['predicted'])
    df_cm = pd.DataFrame(cm,index = colnames, columns=colnames)
    
    cm2 = confusion_matrix(prediction_summary['actual'], prediction_summary['prediction2'])
    df_cm2 = pd.DataFrame(cm2,index = colnames, columns=colnames)
    
    cm3 = confusion_matrix(prediction_summary['actual'], prediction_summary['prediction3'])
    df_cm3 = pd.DataFrame(cm3,index = colnames, columns=colnames)
    
    cm4 = confusion_matrix(prediction_summary['actual'], prediction_summary['prediction4'])
    df_cm4 = pd.DataFrame(cm4,index = colnames, columns=colnames)
     
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu",annot_kws={"size": 16, "color":'r'}, ax=axes[0,0]).set_title('prediction')
    sn.heatmap(df_cm2, annot=True, fmt="d", cmap="YlGnBu",annot_kws={"size": 16, "color":'r'}, ax=axes[0,1]).set_title('prediction2')
    sn.heatmap(df_cm3, annot=True, fmt="d", cmap="YlGnBu",annot_kws={"size": 16, "color":'r'}, ax=axes[1,0]).set_title('prediction3')
    sn.heatmap(df_cm4, annot=True, fmt="d", cmap="YlGnBu",annot_kws={"size": 16, "color":'r'}, ax=axes[1,1]).set_title('prediction4')


def plot_confidence_matrix(prediction_summary, actuals):
    """
    likethe confusion matrix but plots average confidence for each pair

    Parameters
    ----------
    prediction_summary : TYPE
        DESCRIPTION.
    actuals : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
        
    confidences = prediction_summary[['actual','predicted','confidence']]
    avg_conf = confidences.groupby(['actual','predicted']).agg({'confidence': 'mean'})
    
    conf_matrix = pd.DataFrame(index = sorted(list(set(actuals))), columns=sorted(list(set(actuals))))
    avg_conf.reset_index(inplace=True)
    for ind in avg_conf.index:
        conf_matrix[avg_conf['predicted'][ind]][avg_conf['actual'][ind]] = avg_conf['confidence'][ind]
        
    conf_matrix.fillna(0, inplace=True)
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    sn.heatmap(conf_matrix, annot=True, cmap="YlGnBu",annot_kws={"size": 16, "color":'r'}).set_title('mean confidence')
