import time
from contextlib import redirect_stdout
from preprocess import *
from plotting import *
from io import *
from gp import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle

def merge_two_dicts(x, y):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y : type
        Description of parameter `y`.

    Returns
    -------
    type
        Description of returned object.

    """
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def buildModel(params):
    """Short summary.

    Parameters
    ----------
    N_class : type
        Description of parameter `N_class`.

    Returns
    -------
    type
        Description of returned object.

    """

    #build the model
    #basic model #1
    if params['timeDistr']:
        model = keras.Sequential()
        model.add(layers.GRU(params['nGRU'], activation='sigmoid', return_sequences=True))
        model.add(layers.Dropout(params['dropout'], seed=params['randSeed']))
        model.add(layers.BatchNormalization())
        model.add(layers.TimeDistributed(layers.Dense(params['Nclass'], activation='softmax'))) #note -- not functional yet!
    else:
        model = keras.Sequential()
        for i in np.arange(params['nGRU']):
            model.add(layers.GRU( params['nNeurons'], activation='sigmoid'))
            model.add(layers.Dropout(params['dropout'], seed=params['randSeed']))
            model.add(layers.BatchNormalization())
        model.add(layers.Dense(params['Nclass'], activation='softmax'))
    # look at single band in every band! what bands tell us about what classes?

    #basic model #2
#    model = keras.Sequential()
#    model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
#    model.add(layers.Masking(mask_value=0.))
#    model.add(layers.LSTM(paramsNN['nGRU'], return_sequences=True, dropout=paramsNN['dropout']))
#    model.add(layers.Dropout(paramsNN['dropout'], seed=paramsNN['randSeed']))
#    model.add(layers.BatchNormalization())
#    model.add(layers.GRU(paramsNN['nGRU'], activation='sigmoid', return_sequences=False, dropout=paramsNN['dropout']))
#    model.add(layers.Dropout(paramsNN['dropout'], seed=paramsNN['randSeed']))
#    model.add(layers.BatchNormalization())
#    model.add(layers.Dropout(paramsNN['dropout'], seed=paramsNN['randSeed']))
#    model.add(layers.Dense(N_class, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, params

tf.config.optimizer.set_jit(True)

def runRNN(params={}, metafile='./nongp_3k_truthcatalog.txt', testIDs=[], savepath='/Users/agagliano/Documents/Research/HostClassifier/packages/phast/', outputfn=''):
    """The code to run the (very basic) RNN model. Can be run using padding only or GP+padding.

    Parameters
    ----------
    params : dictionary
        The parameters used for the run. An example is as follows:
        params = {'pad': True, 'GP': False,
        'band_stack': 'g', 'bands':'ugrizY','genData':True}

        pad (bool)       : Pad and use one-band inputs (GP param must be False)
        GP  (bool)       : GP-interpolate all bands and pad the tail edge of each LC (pad param must be False)
        band_stack (str) : The single passband to use for the pad-only model.
        bands (str)      : A string listing all passbands, used in the GP model (default is ugrizY)
        genData (bool)   : Re-generates the stacked arrays for input -- if False, reads them from filepath.
        plot (bool)      : Whether to generate a confusion matrix for the model accuracy on the test set.
        verbose (bool)   : Whether to generate output text during training and validation of the RNN.

    metafile : string
        The file containing the metadata for all transients (IDs, class)
    testIDs : array
        An array of the transient IDs contained in the test set. Used to split the data into train/test split.
    savepath : string
        The path to save info on the model architecture and training/validation accuracy during training.
    outputfn : string
        The filename to save info.

    Returns
    -------
    None
        Doesn't return anything!

    """
    CM_dict = {'u':'Blues', 'g':'Oranges', 'r':'Greens', 'i':'Reds', 'z':'Purples', 'Y':'copper'}

    # ts stores the time in seconds
    ts = int(time.time())

    #all the pre-processing steps!
    if params['genData']:
        fullDF = read_in_LC_data(metafile, raw_LC_path, format='SNANA')
        fullDF = shift_lc(fullDF)
        fullDF = correct_time_dilation(fullDF)
        #fullDF = cut_lc(fullDF)
        fullDF = calc_abs_mags(fullDF)
        fullDF = correct_extinction(fullDF, wvs)
    else:
        #checkpoint -- load saved file
        fullDF = pd.read_csv("/Users/agagliano/Documents/Research/HostClassifier/data/DFwithFirstGPModel.tar.gz")
        fullDF['Flux'] = [np.array(x[1:-1].split()).astype(np.float64) for x in fullDF['Flux']]
        fullDF['Flux_Err'] = [np.array(x[1:-1].split()).astype(np.float64)  for x in fullDF['Flux_Err']]
        fullDF['T'] = [np.array(x[1:-1].split()).astype(np.float64)  for x in fullDF['T']]
        fullDF['MJD'] = [np.array(x[1:-1].split()).astype(np.float64)  for x in fullDF['MJD']]
        stackedInputs = pd.read_pickle('/Users/agagliano/Documents/Research/HostClassifier/data/GP_1644294157.pkl')
    fullDF, encoding_dict = encode_classes(fullDF)
    N_class = len(np.unique(list(encoding_dict.keys())))
    params['Nclass'] = N_class

    #define a stackedInputs_test and a stackedInputs_train here
    fullDF_train = fullDF[~fullDF['CID'].isin(testCIDs)]
    fullDF_test = fullDF[fullDF['CID'].isin(testCIDs)]
    if params['pad']:
        stackedInputs_test = stackInputs(fullDF_test, params['band_stack'])
        stackedInputs_train = stackInputs(fullDF_train, params['band_stack'])
    elif params['GP']:
        if params['genData']:
            #stackedInputs = getGPLCs(fullDF, saveDir=savepath,
            #                    bands='ugrizY', ts=ts, fn='firstGPSet')
            #try out the classification with edge padding
            stackedInputs = gp_withPad(fullDF, savepath=savepath,
            plotpath='./', bands=params['bands'], Nstp=params['Nstp'], ts=ts, fn='GPSet')

        stackedInputs_test = {key: stackedInputs[key] for key in testCIDs}
        stackedInputs_train = {key: stackedInputs[key] for key in set(stackedInputs.keys()) - set(testCIDs)}

    fullDF_train = fullDF_train.set_index('CID').loc[list(stackedInputs_train.keys())].reset_index()
    fullDF_test = fullDF_test.set_index('CID').loc[list(stackedInputs_test.keys())].reset_index()

    #set up the train and test sets
    X_train = np.swapaxes(list(stackedInputs_train.values()), 1, 2)
    X_test = np.swapaxes(list(stackedInputs_test.values()), 1, 2)
    y_train = fullDF_train['Type_ID'].values
    y_test = fullDF_test['Type_ID'].values

    #compile the model with specific choices for the log
    model, params = buildModel(params)

    #write all model info to file!
    #write the training and the model summary to file
    if params['verbose']:
        writeModel(savepath, outputfn, model, params, X_train, y_train, X_test, y_test)
    else:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params['batch_size'], epochs=params['nepochs'], verbose=2)

    if params['plot']:
        fn=outputfn
        if params['pad']:
            fn = outputfn+"_%s"%band
        makeCM(model, X_train, X_test, y_train, y_test, encoding_dict, fn=fn, ts=ts, plotpath=savepath+'/plots/',c=CM_dict[band])
