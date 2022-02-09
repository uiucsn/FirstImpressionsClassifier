import time
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from preprocess import *
from gp import *
from rnn import *
from plotting import *

import pickle

tf.config.optimizer.set_jit(True)

inDir = '/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/'
outDir = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/'
raw_LC_path = inDir + "/lcs_notrigger"
metafile = inDir + "/nongp_3k_truthcatalog.txt"
metafile_Test = inDir + "sn_3k_list_unblindedtestSet.txt" #get the CIDs of the train and test sets
testCIDs = pd.read_csv(metafile_Test, delim_whitespace=True)['CID'].values

params = {'pad': True, 'GP': False,
    'band_stack': 'g', 'bands':'ugrizY','genData':True,
    'verbose':True, 'plot':True,'Ntsp':100}

paramsNN = {'batch_size':128,
'nepochs':100,
'nNeurons':50,
'nGRU':1,
'dropout':0.2,
'randSeed':42,
'timeDistr':False}

params = merge_two_dicts(params, paramsNN)

runRNN(params=params,
        metafile=metafile,
        testIDs=testCIDs,
        savepath=outDir,
        outputfn='rnn_testing_')
