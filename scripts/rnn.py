from contextlib import redirect_stdout
from lcIO import *
from preprocess import *
from plotting import *
from gp import *
import tensorflow as tf
from tensorflow import keras
from helper_functions import *
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LSTM
import time

def buildModel(params):
    """A function to construct the classification RNN. By default, we use a GRU with one dropout and one batch normalization component.

    Parameters
    ----------
    params : dictionary
        Contains the full params for the run. The ones used by the NN are:
            nGRU (int)       : Number of GRU layers. Each GRU layer has a dropout and a batch
                               normalization component.
            nNeurons (int)   : Number of neurons in each GRU layer.
            randSeed (int)   : Random seed for re-producing the individual layer configurations.
            Nclass (int)     : Number of transient classes (for the dense layer).
            dropout (float)  : Dropout fraction in each GRU layer.
            timeDistr (bool) : Outputs classification predictions at each time step (valuable for real-time classification!)

    Returns
    -------
    model
        The keras RNN model.

    """
    inputs = layers.Input(shape=(params['Ntstp'], params['Nfeatures']))
    hidden = layers.Masking(mask_value=0.)(inputs)
    hidden = layers.LSTM(params['nNeurons'], activation=params['activation'], return_sequences=False, dropout=params['dropout'], recurrent_dropout=params['dropout'])(hidden)
    class_output = layers.Dense(params['Nclass'], activation='softmax', name='class_output')(hidden)
    adm = keras.optimizers.Adam(learning_rate=paramsNN['learning_rate'])
    model = models.Model(inputs=inputs, outputs=class_output)
    model.compile(optimizer=adm, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def writeModel(params, model,  X_train, y_train, X_test, y_test,  savepath='./', outputfn='rnn_testing_', ts='0000000'):
    """Train the keras model.

    Parameters
    ----------
    params : dictionary
        A dictionary of full params for the run.
    model : keras Model
        The compiled keras model for training.
    X_train : numpy matrix
        The features of the training set.
    y_train : numpy array
        The target encoded classifications of the training set.
    X_test : numpy matrix
        The features of the test set.
    y_test : numpy array
        The target encoded classifications of the test set.
    savepath : str
        A string for the path where the plot will be saved.
    outputfn : str
        The filename where the model training history will be written.
    ts : str
        The timestamp of the run (used to save unique runs).
    Returns
    -------
    type
        Description of returned object.

    """
    textPath = savepath + '/text/'
    textfile = open(textPath + "/" + outputfn + "_%s_%s.txt"%(ts, params['bands']), "at")
    for key in params.keys():
        textfile.write("{} = {}\n".format(key, params[key]))
    textfile.write("Training:\n")

    #save checkpoint weights of the model each time the validation accuracy improves
    mc = ModelCheckpoint(savepath+'/models/Model_%s_CheckpointWeights.sav'%outputfn, monitor='val_accuracy', mode='max', verbose=1, save_weights_only=True, save_best_only=True)

    with redirect_stdout(textfile):
        weights = np.ones(np.shape(X_train[:, :, 0]))
        weights[(X_train[:, :, 0][:, -1] < 3)] = 10
        weights[(X_train[:, :, 0][:, -1] > 3) & (X_train[:, :, 0][:, -1] < 15)] = 5

        #the class-specific weights (only applicable in the spectroscopic re-training stage)
        weights[y_train == 0] *= params['class_weight'][0]
        weights[y_train == 1] *= params['class_weight'][1]
        weights[y_train == 2] *= params['class_weight'][2]

        #compress -- not weighting doing in time anymore
        weights = weights[:, 0]

        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params['batch_size'], epochs=params['nepochs'], verbose=2, callbacks=[mc], sample_weight=weights)#class_weight=params['class_weight'])#sample_weight=weights)
        model.summary()

    #load the best version in terms of the validation accuracy, and save it!
    model.load_weights(savepath+'/models/Model_%s_CheckpointWeights.sav'%outputfn)
    model.save(savepath+'/models/Model_%s.sav'%outputfn)
    textfile.close()
