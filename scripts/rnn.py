import time
from contextlib import redirect_stdout
from lcIO import *
from preprocess import *
from plotting import *
from gp import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle

def merge_dicts(x):
    """A quick function to combine dictionaries.

    Parameters
    ----------
    x : list or array-like
        A list of dictionaries to merge.

    Returns
    -------
    dict
        The merged dictionary.

    """
    z = x[0].copy()   # start with keys and values of x
    for y in x[1:]:
        z.update(y)    # modifies z with keys and values of y
    return z

def buildModel(params):
    """A function to construct the classification RNN. By default, we use a GRU with one dropout and one batch normalization component.

    Parameters
    ----------
    params : A dictionary of full params for the run. The ones used by the NN are:

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

    return model

tf.config.optimizer.set_jit(True)

def runRNN(params={}, metafile='./nongp_3k_truthcatalog.txt',
        testIDs=[], datapath='./', savepath='/Users/agagliano/Documents/Research/HostClassifier/packages/phast/',
        outputfn=''):
    """The code to run the (very basic) RNN model. Can be run using padding only or GP+padding.

    Parameters
    ----------
    params : dictionary
        The parameters used for the run, GP, and NN. The dictionary includes:

        pad (bool)       : Pad and use one-band inputs (GP param must be False)
        GP  (bool)       : GP-interpolate all bands and pad the tail edge of each LC (pad param must be False)
        band_stack (str) : The single passband to use for the pad-only model.
        bands (str)      : A string listing all passbands, used in the GP model (default is ugrizY)
        genData (bool)   : Re-generates the stacked arrays for input -- if False, reads them from filepath.
        Ntsp (int)       : The number of time steps to use for interpolation over the flux in each band (only used if GP=True).
        plot (bool)      : Whether to generate a confusion matrix for the model accuracy on the test set.
        verbose (bool)   : Whether to generate output text during training and validation of the RNN.
        #TODO: Add in the other parameters for the NN and the GP!

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
        fullDF = read_in_LC_data(metafile, datapath, format='SNANA', save=True, ts=ts, savepath=savepath + '/data/')
        fullDF = shift_lc(fullDF)
        fullDF = correct_time_dilation(fullDF)
        if params['GP']:
            fullDF = gp_withPad(fullDF, savepath=savepath+'/data/',
            plotpath=savepath+'/plots/GP/', bands=params['bands'], Ntstp=params['Ntstp'], ts=ts, fn='GPSet')
        fullDF = calc_abs_mags(fullDF)
        fullDF = correct_extinction(fullDF, wvs)

        fullDF, encoding_dict = encode_classes(fullDF)
        N_class = len(np.unique(list(encoding_dict.keys())))
        params['Nclass'] = N_class

        df_test = fullDF.sample(frac=params['testFrac'])
        df_train = fullDF[~fullDF['CID'].isin(df_test['CID'])]
        #this whole thing is outdated now, just split the dataset up into train and test
        #define a stackedInputs_test and a stackedInputs_train here
        #fullDF_train = fullDF[~fullDF['CID'].isin(testIDs)]
        #fullDF_test = fullDF[fullDF['CID'].isin(testIDs)]

        stackedInputs_test = stackInputs(df_test, params)
        stackedInputs_train = stackInputs(df_train, params)
        #stackedInputs = stackInputs(fullDF, params)

        #stackedInputs_test = {key: stackedInputs[key] for key in testIDs}
        #stackedInputs_train = {key: stackedInputs[key] for key in set(stackedInputs.keys()) - set(testIDs)}
    else:
        #checkpoint -- load saved file
        # TODO -- remove hardcoded path!
        fullDF = pd.read_csv("/Users/agagliano/Documents/Research/HostClassifier/data/DFwithFirstGPModel.tar.gz")
        fullDF['Flux'] = [np.array(x[1:-1].split()).astype(np.float64) for x in fullDF['Flux']]
        fullDF['Flux_Err'] = [np.array(x[1:-1].split()).astype(np.float64)  for x in fullDF['Flux_Err']]
        fullDF['T'] = [np.array(x[1:-1].split()).astype(np.float64)  for x in fullDF['T']]
        fullDF['MJD'] = [np.array(x[1:-1].split()).astype(np.float64)  for x in fullDF['MJD']]
        stackedInputs = pd.read_pickle('/Users/agagliano/Documents/Research/HostClassifier/data/GP_1644294157.pkl')

    #re-order to make sure the X and y sets match
    df_train = df_train.set_index('CID').loc[list(stackedInputs_train.keys())].reset_index()
    df_test = df_test.set_index('CID').loc[list(stackedInputs_test.keys())].reset_index()

    #set up the train and test sets
    X_train = np.swapaxes(list(stackedInputs_train.values()), 1, 2)
    X_test = np.swapaxes(list(stackedInputs_test.values()), 1, 2)
    y_train = df_train['Type_ID'].values
    y_test = df_test['Type_ID'].values

    #compile the model with specific choices for the log
    model = buildModel(params)

    #write all model info to file!
    #write the training and the model summary to file
    if params['verbose']:
        writeModel(params, model, X_train, y_train, X_test, y_test, savepath, outputfn, ts)
    else:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params['batch_size'], epochs=params['nepochs'], verbose=2)

    if params['plot']:
        fn=outputfn
        if not params['GP']:
            fn = outputfn+"_%s"%params['band_stack']
            col = CM_dict[params['band_stack']]
        else:
            col = 'Reds'
        makeCM(model, X_train, X_test, y_train, y_test, encoding_dict, fn=fn, ts=ts, plotpath=savepath+'/plots/',c=col)
