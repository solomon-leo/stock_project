import pandas as pd 
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from scipy.signal import resample_poly, filtfilt, iirfilter, detrend, butter
import matplotlib.pyplot as plt
from keras.models import Model

class model_wrap:
    # Initialize model
    def __init__(self, model: keras.models.Model, transform=None, invert_transform=None, scalar=None):
        self.model = model
        self.transform = transform
        self.invert_transform = invert_transform
        self.scaler = scalar

    # Fit model and return fit model object
    def fit(self, X, y, optimizer, loss, epochs=20):
        
        # Check transform flag
        if self.transform is not None:
            y = self.transform(y)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        # compile and fit model
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(X, y, epochs=epochs)

        return self.model
    
    # Predict using model
    def predict(self, X, y):
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        ymin, ymax = get_ends(y)
        y_pred = self.model.predict(X)

        if self.invert_transform is not None:
            y_pred = self.invert_transform(y_pred, ymin, ymax)

        return y_pred
    
# Get the min and max of any data shape
def get_ends(y):
    max = []
    min = []

    for col in range(y.shape[1]):
        max.append(np.max(y[:, col]))
        min.append(np.min(y[:, col]))

    return min, max

# Divide everything by the max value
def transform(y):
    res = np.zeros_like(y)
    for col in range(y.shape[1]):
        res[:, col] = (y[:, col])/np.max(y[:, col])

    return res

def invert_transform(y, ymin, ymax):
    res = np.zeros_like(y)
    for col in range(y.shape[1]):
        res[:, col] = (y[:, col])*ymax[col]

    return res

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))