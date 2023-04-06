import pandas as pd
import numpy as np
import settings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



def preprocess(data, oneHot=False):
    #removing redundant columns
    newData = data.drop(labels=settings.redundantFeatures, axis=1)

    #making labels numerical - 1 for >50K, 0 for <=50K or unknown
    newData["class"] = newData["class"].map(lambda x: 1 if(x == ">50K") else 0)

    if oneHot:
        #converting dataframe to one-hot representation with one column for each possible value of each categorical attribute
        newData = pd.get_dummies(newData, columns=settings.categoricalFeatures)
        
    return newData




def preprocessDT(data):
    return preprocess(data, oneHot=True)




def preprocessNB(data):
    processed = preprocess(data)

    #encoding categorical features
    processed[settings.categoricalFeatures] = processed[settings.categoricalFeatures].apply(LabelEncoder().fit_transform)
    
    return processed


def preprocessKNN(data):
    processed = preprocess(data, oneHot=True)

    #normalising categorical attributes
    oneHotCategorical = list(set(processed.columns.values.tolist()) - set(settings.numericalFeatures))    #getting list of one-hot encoded categorical column names
    oneHotCategorical.remove("class")
    weight = 0.5 * len(settings.numericalFeatures)/len(settings.categoricalFeatures)

    processed[oneHotCategorical] = processed[oneHotCategorical].apply(lambda x: x*weight)
    #processed.head()

    return processed


def scaleNumericalFeatures(data, scaler=None):
    '''
        Does scaling of numerical features of the passed data parameter, using the scaler parameter.
        If no scaler is given, a new scaler will be fit to the given data.

        Returns: pandas dataframe with scaled numerical features, scaler which was used to scale the data.
    '''
    if scaler == None:
        #normalising numerical attributes
        scaler = MinMaxScaler()
        scaler.fit(data[settings.numericalFeatures])
        #print(scaler.data_max_)

    d = data.copy()
    d[settings.numericalFeatures] = scaler.transform(d[settings.numericalFeatures])
    #data.head()

    return d, scaler



def customScorer(truths, preds):
    #if correctly predicted 0, return 0
    #if incorrectly predicted 0, return 0
    #if correctly predicted 1, return (0.1 * 980) - 10
    #if incorrectly predicted 1, return -(0.05 * 310) - 10

    convTruths = truths.tolist()
    score = 0
    
    for i in range(len(convTruths)):
        if preds[i] == 0:
            continue

        if convTruths[i] == 1:
            score += (0.1 * 980) - 10

        else:
            score += -(0.05 * 310) - 10

    return score
    