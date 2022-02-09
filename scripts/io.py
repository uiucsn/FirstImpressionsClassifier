from preprocess import *
import time
from contextlib import redirect_stdout
from gp import *
import tensorflow as tf
from plotting import *
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle

def writeModel(savepath='./', outputfn='rnn_testing_', model=None, params, X_train=None, y_train=None, X_test=None, y_test=None):
    """Short summary.

    Parameters
    ----------
    savepath : type
        Description of parameter `savepath`.
    outputfn : type
        Description of parameter `outputfn`.
    model : type
        Description of parameter `model`.
    X_train : type
        Description of parameter `X_train`.
    y_train : type
        Description of parameter `y_train`.
    X_test : type
        Description of parameter `X_test`.
    y_test : type
        Description of parameter `y_test`.

    Returns
    -------
    type
        Description of returned object.

    """
    textPath = savepath + '/text/'
    if params['pad']:
        textfile = open(textPath + "/" + outputfn + "_%s_%s.txt"%(ts, params['band_stack']), "wt")
    else:
        textfile = open(textPath + "/" + outputfn + "_%s_%s.txt"%(ts, params['band_stack']), "wt")
    for key in params.keys():
        textfile.write("{} = {}\n".format(key, params[key]))
    textfile.write("Training:\n")
    with redirect_stdout(textfile):
        # fit the data!
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params['batch_size'], epochs=params['nepochs'], verbose=2)
        model.summary()
    textfile.close()
