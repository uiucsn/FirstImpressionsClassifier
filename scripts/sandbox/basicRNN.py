from preprocess import *
import time
from contextlib import redirect_stdout
from gp import *
import tensorflow as tf
from plotting import *
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle

def merge_two_dicts(x, y):
    """A quick function to combine dictionaries.

    Parameters
    ----------
    x : dict
        Dictionary 1.
    y : dict
        Dictionary 2.

    Returns
    -------
    dict
        The merged dictionary.

    """
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def buildModel(N_class, params, X_train):
    """A function to construct the classification RNN.

    Parameters
    ----------
    N_class : type
        Description of parameter `N_class`.
    N_class : type
        Description of parameter `N_class`.
    N_class : type
        Description of parameter `N_class`.

    Returns
    -------
    model
        The keras RNN model.
    params
        Description of returned object.

    """
    #build the model
    #basic model #1
    model = keras.Sequential()
    model.add(layers.GRU( params['nGRU'], activation='sigmoid'))
    model.add(layers.Dropout(params['dropout'], seed=params['randSeed']))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(N_class, activation='softmax'))
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

raw_LC_path = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/lcs_notrigger"
metafile = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/nongp_3k_truthcatalog.txt"
metafile_Test = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/sn_3k_list_unblindedtestSet.txt" #get the CIDs of the train and test sets
testCIDs = pd.read_csv(metafile_Test, delim_whitespace=True)['CID'].values
savepath = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/plots'

params = {'pad': True, 'GP': False,
'band_stack': 'g', 'bands':'ugrizY','genData':True}

CM_dict = {'u':'Blues', 'g':'Oranges', 'r':'Greens', 'i':'Reds', 'z':'Purples', 'Y':'copper'}

for band in 'ugrizY':
    # ts stores the time in seconds
    ts = int(time.time())

    params['band_stack'] = band
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

    #define a stackedInputs_test and a stackedInputs_train here
    fullDF_train = fullDF[~fullDF['CID'].isin(testCIDs)]
    fullDF_test = fullDF[fullDF['CID'].isin(testCIDs)]
    if params['pad']:
        stackedInputs_test = stackInputs(fullDF_test, params['band_stack'])
        stackedInputs_train = stackInputs(fullDF_train, params['band_stack'])
    elif params['GP']:
        if params['genData']:
            stackedInputs = getGPLCs(fullDF, plotpath='/Users/agagliano/Documents/Research/HostClassifier/plot/',
                                savepath='/Users/agagliano/Documents/Research/HostClassifier/plot/',
                                bands=params['bands'], ts=ts, fn='firstGPSet')
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
    model, params = buildModel(N_class, params, X_train)

    #write all model info to file!
    #write the training and the model summary to file
    textPath = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/text/'
    textfile = open(textPath + "/Model_%i_%s.txt"%(ts, band), "wt")
    for key in params.keys():
        textfile.write("{} = {}\n".format(key, params[key]))
    textfile.write("Training:\n")

    with redirect_stdout(textfile):
        # fit the data!
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params['batch_size'], epochs=params['nepochs'], verbose=2)
        model.summary()
    textfile.close()

    makeCM(model, X_train, X_test, y_train, y_test, encoding_dict, fn='gp_rnn_LSSTexp_%s'%band, ts=ts, plotpath='/Users/agagliano/Documents/Research/HostClassifier/packages/phast/plots/',c=CM_dict[band])
