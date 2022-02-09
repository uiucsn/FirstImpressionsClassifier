from preprocess import *
import time
from contextlib import redirect_stdout
from gp import *
import tensorflow as tf
from plotting import *
from rnn import *
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle
from sklearn.model_selection import GridSearchCV

tf.config.optimizer.set_jit(True)

raw_LC_path = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/lcs_notrigger"
metafile = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/nongp_3k_truthcatalog.txt"
metafile_Test = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/sn_3k_list_unblindedtestSet.txt" #get the CIDs of the train and test sets
testCIDs = pd.read_csv(metafile_Test, delim_whitespace=True)['CID'].values
savepath = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/plots'

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
    stackedInputs = pd.read_pickle('/Users/agagliano/Documents/Research/HostClassifier/data/GP_1644294157.pkl')
fullDF, encoding_dict = encode_classes(fullDF)
N_class = len(np.unique(list(encoding_dict.keys())))

#define a stackedInputs_test and a stackedInputs_train here
fullDF_train = fullDF[~fullDF['CID'].isin(testCIDs)]
fullDF_test = fullDF[fullDF['CID'].isin(testCIDs)]
if params['pad']:
    stackedInputs_test = stackInputs(fullDF_test, band_stack)
    stackedInputs_train = stackInputs(fullDF_train, band_stack)
elif params['GP']:
    if params['genData']:
        stackedInputs = getGPLCs(fullDF, plotpath='/Users/agagliano/Documents/Research/HostClassifier/plot/',
                            savepath='/Users/agagliano/Documents/Research/HostClassifier/plot/',
                            bands='ugrizY', ts=ts, fn='firstGPSet')
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
from keras.wrappers.scikit_learn import KerasClassifier

#model = buildModel(N_class, params, X_train)
model = KerasClassifier(build_fn=buildModel)

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    n_jobs=-1,
                    cv=3,
                    return_train_score=True,
                    scoring=['accuracy'],refit='accuracy')
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#write all model info to file!
#write the training and the model summary to file
textPath = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/text/'
textfile = open(textPath + "/Model_%i.txt"%ts, "wt")
for key in params.keys():
    textfile.write("{} = {}\n".format(key, params[key]))
textfile.write("Training:\n")
with redirect_stdout(textfile):
    # fit the data!
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params['batch_size'], epochs=params['nepochs'], verbose=2)
    model.summary()
textfile.close()

np.shape(X_train)
makeCM(model, X_train, X_test, y_train, y_test, encoding_dict, fn='gp_rnn_LSSTexp', ts=ts)
