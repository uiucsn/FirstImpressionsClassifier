from preprocess import *
import time
from contextlib import redirect_stdout
from gp import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

raw_LC_path = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/lcs_notrigger"
metafile = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/nongp_3k_truthcatalog.txt"
metafile_Test = "/Users/agagliano/Documents/Research/HostClassifier/data/3k_NONGP/sn_3k_list_unblindedtestSet.txt" #get the CIDs of the train and test sets
testCIDs = pd.read_csv(metafile_Test, delim_whitespace=True)['CID'].values
savepath = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/plots'

pad = False
GP = True
band_stack = 'g'
bands = 'ugrizY'
N_bands = len(bands)
#timestamp

# ts stores the time in seconds
ts = int(time.time())

#all the pre-processing steps!
fullDF = read_in_LC_data(metafile, raw_LC_path, format='SNANA')
fullDF = shift_lc(fullDF)
fullDF = correct_time_dilation(fullDF)
#fullDF = cut_lc(fullDF)
fullDF = calc_abs_mags(fullDF)
fullDF = correct_extinction(fullDF, wvs)
fullDF, encoding_dict = encode_classes(fullDF)

N_class = len(np.unique(list(encoding_dict.keys())))

fullDF_train = fullDF[~fullDF['CID'].isin(testCIDs)]
fullDF_test = fullDF[fullDF['CID'].isin(testCIDs)]

if GP:
    fullDF = getGPLCs(fullDF, savepath=savepath+"/GP/", num_bands=N_bands) ## not defined yet
    #define a stackedInputs_test and a stackedInputs_train here

#just do in a single band for now!
#if pad:
#    stackedInputs_test = stackInputs(fullDF_test, band_stack)
#    stackedInputs_train = stackInputs(fullDF_train, band_stack)
fullDF.to_csv("/Users/agagliano/Documents/Research/HostClassifier/data/DFwithFirstGPModel.tar.gz",index=False)
#started at 1145am
#ended at...

#remove this step later -- dropping those that didn't make our quality cuts
fullDF_train = fullDF_train[fullDF_train['CID'].isin(list(stackedInputs_train.keys()))]
fullDF_test = fullDF_test[fullDF_test['CID'].isin(list(stackedInputs_test.keys()))]

#set up the train and test sets
X_train = np.swapaxes(list(stackedInputs_train.values()), 1, 2)
y_train = fullDF_train['Type_ID'].values

X_test = np.swapaxes(list(stackedInputs_test.values()), 1, 2)
y_test = fullDF_test['Type_ID'].values

#build the model
model = keras.Sequential()
#model.add(layers.GRU(100, activation='sigmoid', return_sequences=True))
#model.add(layers.Dropout(0.2, seed=42))
#model.add(layers.BatchNormalization())
model.add(layers.GRU(100, activation='sigmoid'))
#model.add(layers.Dropout(0.2, seed=42))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(6, activation='softmax')) #the number of classes

batch_size = 128

#compile the model with specific choices for the log
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#write all model info to file!
textPath = '/Users/agagliano/Documents/Research/HostClassifier/packages/phast/text/'
textfile = open(textPath + "/Model_%i.txt"%ts, "wt")

textfile.write("Pad input arrays? %s\n"%pad)
textfile.write("Use GP interpolation? %s\n"%GP)
textfile.write("What bands input? %s\n\n"%band_stack)
textfile.write("Training:\n")

#write the training and the model summary to file
with redirect_stdout(textfile):
    # fit the data!
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=100, verbose=2)
    model.summary()
textfile.close()

# make predictions
predictions = model.predict(X_test)
predictDF = pd.DataFrame(data=predictions, columns=encoding_dict.values())
predictDF['PredClass'] = predictDF.idxmax(axis=1)
predictDF['TrueClass'] = [encoding_dict[x] for x in y_test]
accTest = np.sum(predictDF['PredClass'] == predictDF['TrueClass'])/len(predictDF)*100

#create confusion matrix
CM = confusion_matrix(predictDF['TrueClass'], predictDF['PredClass'], normalize='true')
fig = plt.figure(figsize=(10.0, 8.0), dpi=300) #frameon=false
df_cm = pd.DataFrame(CM, columns=np.unique(predictDF['PredClass'].values), index = np.unique(predictDF['PredClass'].values))
df_cm.index.name = 'True Label'
df_cm.columns.name = 'Predicted Label'

#plot it here:
plt.figure(figsize = (10,7))
sns.set(font_scale=2)
g = sns.heatmap(df_cm, cmap="Reds", annot=True, fmt=".2f", annot_kws={"size": 30}, linewidths=1, linecolor='black', cbar_kws={"ticks": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, vmin=0.29, vmax=0.91)# font size
g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
g.set_yticklabels(g.get_yticklabels(), fontsize = 20)
g.set_title("Test Set, Accuracy = %.2f%%"%accTest)
plt.savefig(savepath + "/vanilla_rnn_LSSTexp_%i.png"%ts, dpi=200, bbox_inches='tight')

########################### sandbox testing below! ###########################
