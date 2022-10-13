import os
import time
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from lcIO import *
from preprocess import *
from gp import *
from rnn import *
from plotting import *
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
from collections import Counter

np_config.enable_numpy_behavior()
tf.config.optimizer.set_jit(True)

outputDir = '/Users/alexgagliano/Documents/Research/HostClassifier/packages/phast/'
outputfn = "ZTFSims_100Timestep"
ts = int(time.time())

paramsRun = {
    'GPMethod': '100Timestep', # whether GP interpolation was used
    'bands':'XY', # the transient light curve bands (in this case, X and Y correspond to ZTF g and r)
    'verbose':True,
    'plot':True,
    'MWEBV':True, # includes milky way extinction values in processed array
    'z':True,  # includes photometric redshift in processed array
    'hostPhot':True,  # includes host galaxy photometry in processed array
    'class_weight':{0:1, 1:1, 2:1},  # the weights to give to each class -- they're even here, in the first training stage
    'Nfeatures': 2*len(params['bands'])+1 #all bands, + error, + time
    'outputfn':outputfn
}

paramsNN = {
    'batch_size':16, #the size of the batch samples for training
    'nepochs':100, #number of epochs used for training
    'nNeurons':60, #number of neurons per LSTM layer
    'nLayers':1, # the number of recurrent layers to use
    'dropout':0.0, #the fraction of the neurons to dropout in training the NN
    'randSeed':3, #random seed to reproduce the training of the RNN
    'gate': 'lstm', # the rnn gate used
    'activation':'sigmoid', # activation function for the rnn layer
    'learning_rate':1.e-3 # the learning rate
}

paramsGP = {
    "mean": np.zeros(len(paramsRun['bands'])), # the mean model for the interpolation GP -- 0 by default
    "log_scale": np.log(100.0), # the scale factor for the
    "log_diagonal": np.zeros(len(paramsRun['bands'])),
    "off_diagonal": np.zeros(((len(paramsRun['bands']) - 1) * len(paramsRun['bands'])) // 2),
    "log_jitter": np.zeros(len(paramsRun['bands'])),
}

params = merge_dicts([paramsRun, paramsNN, paramsGP])

textPath = outputDir + '/text/'
textfile = open(textPath + "/" + outputfn + "_%s_%s.txt"%(ts, params['bands']), "at")

df_train = pd.read_json(inDir + "/data/ZTFBTS_Sliced_Train_Full_Shuffled.json")
df_test = pd.read_json(inDir + "/data/ZTFBTS_Sliced_Test_Full_Shuffled.json")

X_train, y_train, X_test, y_test = build_rnn_arrays(params, df_train, df_test)

############### plotting the results #####################
model = buildModel(params)
writeModel(params, model,  X_train, y_train, X_test, y_test,  savepath=outputDir, outputfn=outputfn, ts=ts)
model.load_weights(outputDir+'/models/Model_%s_CheckpointWeights.sav'%outputfn)

######### building and training the model ################
phaseDict = {'First3':(0, 3), 'Mid15':(3, 15), 'Last15':(15, 30)}
for name, bounds in phaseDicts.items():
    fig, c_ax = plt.subplots(1,1, figsize = (8, 8))
    #Receiver Operator Characteristic Curves for the 3 phases
    plot_ROC_wCV(model, params, fig.gca(), X, y, bounds, encoding_dict, fnstr=outputfn + "_" + name, plotpath=outputDir+'/plots/', save=True)

    fig, c_ax = plt.subplots(1,1, figsize = (8, 8))
    #Precision-Recall Curves for the 3 phases
    plot_PR_wCV(model, params, fig.gca(), X, y, bounds, encoding_dict, fnstr=outputfn + "_" + name, plotpath=outputDir+'/plots/', save=True)
