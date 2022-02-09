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

tf.config.optimizer.set_jit(True)

#########################################################################################################################################
#########################################################################################################################################
######## NOTE: in the first case (raw stacking), we're correcting for exinction and shifting to absolute magnitude               ########
########       in the SECOND case (GP interp), we're using raw flux (not correcting for extinction and distance). Change this!!  ########
#########################################################################################################################################

inDir = '/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/'
outDir = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/'
raw_LC_path = inDir + "/lcs_notrigger"
metafile = inDir + "/nongp_3k_truthcatalog.txt"
metafile_Test = inDir + "sn_3k_list_unblindedtestSet.txt" #get the CIDs of the train and test sets
testCIDs = pd.read_csv(metafile_Test, delim_whitespace=True)['CID'].values
outputfn = 'rnn_testing'

paramsRun = {
    'GP': True,
    'band_stack': '',
    'bands':'ugrizY',
    'genData':True,
    'verbose':True,
    'plot':True,
    'Ntstp':100
}

paramsNN = {
    'batch_size':128,
    'nepochs':100,
    'nNeurons':50,
    'nGRU':1,
    'dropout':0.2,
    'randSeed':42,
    'timeDistr':False
}

paramsGP = {
    "mean": np.zeros(len(paramsRun['bands'])),
    "log_scale": np.log(100.0),
    "log_diagonal": np.zeros(len(paramsRun['bands'])),
    "off_diagonal": np.zeros(((len(paramsRun['bands']) - 1) * len(paramsRun['bands'])) // 2),
    "log_jitter": np.zeros(len(paramsRun['bands'])),
}

params = merge_dicts([paramsRun, paramsNN, paramsGP])

runRNN(params, metafile,testCIDs, raw_LC_path, outDir, outputfn)
