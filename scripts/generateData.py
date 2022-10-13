import numpy as np
import pandas as pd
import seaborn as sns
import glob
import os
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import time
from contextlib import redirect_stdout
from lcIO import *
from preprocess import *
from imblearn.under_sampling import RandomUnderSampler
from helper_functions import *

#read data frame -- in this case, for the simulated sample
df = pd.read_json("./ZTFSims.json")

#generate a new dataframe, where each LC is represented as a series of segments, one in each row (LCs are segmented every two points)
df_sliced = segment_df(df)

#jitter the trigger phase in order to make the dataset a bit more robust against survey-specific triggering
df_jitter = jitter_lcs(df_sliced, headrand=0.25)

#generate training set
under = RandomUnderSampler(sampling_strategy={'SNIa':10000, 'SNIbc':10000, 'SNII':10000})
df_train_x, df_train_y = under.fit_resample(df_jitter.drop(columns=['Type']), df_jitter['Type'])
df_train = pd.DataFrame(df_train_x, columns=df_jitter.drop(columns=['Type']).columns)
df_train['Type'] = df_train_y

#generate testing set
df_remaining = df_jitter[~df_jitter['CID'].isin(df_train['CID'])]
under = RandomUnderSampler(sampling_strategy={'SNIa':8000, 'SNIbc':500, 'SNII':1500})
df_test_x, df_test_y = under.fit_resample(df_remaining.drop(columns=['Type']), df_remaining['Type'])
df_test = pd.DataFrame(df_test_x, columns=df_remaining.drop(columns=['Type']).columns)
df_test['Type'] = df_test_y

#reset indices
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

#Generate the dictionary of interpolated Gaussian Process light-curves
#Note: this may take a while; it's recommended to do this via batch submission
gen_gp(df_train)
gen_gp(df_test)

#load in and save the GP dictionaries as one
trainDicts = glob.glob("ZTF_Train*.pkl")
testDicts = glob.glob("ZTF_Test*.pkl")

#once for the training set
trainDicts_list = []
for fn in trainDicts:
    trainDicts_list.append(pd.read_pickle(fn))
trainDicts_merged = merge_dicts(trainDicts_list)

#and once for the test set
testDicts_list = []
for fn in testDicts:
    testDicts_list.append(pd.read_pickle(fn))
testDicts_merged = merge_dicts(testDicts_list)

#pad to make sure they're all the same length!
#these have to be done together to ensure train and test sets are all the same shape
train_dict_padded, test_dict_padded = pad_dicts(params, trainDicts_merged, testDicts_merged)

with open('./ZTFSims_TestSet_RealTime0pt2_Full.pkl', 'wb') as f:
    pickle.dump(test_dict_padded, f)
with open('./ZTFSims_TrainSet_RealTime0pt2_Full.pkl', 'wb') as f:
    pickle.dump(train_dict_padded, f)

#add the GP-interpolated light curve segments back to the now-segmented data frame
fullDF_train = add_segments_to_df(params, df_train, trainDicts_merged)
fullDF_train.to_json("ZTFSims_0pt2Model_Train.json")

fullDF_test = add_segments_to_df(params, df_test, testDicts_merged)
fullDF_test.to_json("ZTFSims_0pt2Model_Test.json")
