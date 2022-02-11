from preprocess import *
import time
from contextlib import redirect_stdout
from gp import *
import tensorflow as tf
from plotting import *
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle

def read_in_LC_data(metafile='./metafile.txt', LC_path='./', format='SNANA', save=True, ts='0000000', savepath='./'):
    """Reads in a directory of SNANA-formatted LC files and outputs a pandas
       dataframe of class, redshift, extinction, discovery date, and photometry.
       Borrowed heavily from superraenn (Villar+2019)!

    Parameters
    ----------
    metafile : string
        Description of parameter `metafile`.
    LC_path : string
        Description of parameter `LC_path`.
    format : string
        Description of parameter `format`.
    save : string
        Description of parameter `format`.
    Returns
    -------
    type
        Description of returned object.
    """

    LC_list = []
    input_files = glob.glob(LC_path + "/" + "*.DAT")
    if format == 'SNANA':
        for i, input_file in enumerate(input_files):
            sn_name = input_file.split("/")[-1].replace(".DAT", "").split("_")[-1].replace("SN", "")
            try:
                num = 71
                t, f, filts, err = np.genfromtxt(input_file,
                                            usecols=(1, 4, 2, 5), skip_header=num,
                                            skip_footer=1, unpack=True, dtype=str)
            except:
                with open(input_file) as tempFile:
                    for num, line in enumerate(tempFile, 1):
                         if 'VARLIST' in line:
                            break
                t, f, filts, err = np.genfromtxt(input_file,
                                             usecols=(1, 4, 2, 5), skip_header=num,
                                             skip_footer=1, unpack=True, dtype=str)
            t = np.asarray(t, dtype=float)
            f = np.asarray(f, dtype=float)
            err = np.asarray(err, dtype=float)

            #sn_name = obj_names[i]
            new_LC = pd.DataFrame({'CID':sn_name, 'MJD':[t], 'Flux':[f], 'Flux_Err':[err], 'Filter':[filts]})
            LC_list.append(new_LC)
    else:
        raise ValueError('Sorry, you need to specify an input format.')
    LCs = pd.concat(LC_list, ignore_index=True)
    metatable = pd.read_csv(metafile, delim_whitespace=True)
    metatable['CID'] = np.int64(metatable['CID'])
    LCs['CID'] = np.int64(LCs['CID'])
    df = LCs.merge(metatable)
    if save:
        df.to_csv(savepath + "/allRawData_%i.tar.gz"%ts)
    return df

def writeModel(params, model,  X_train, y_train, X_test, y_test,  savepath='./', outputfn='rnn_testing_', ts='0000000'):
    """Short summary.

    Parameters
    ----------
    savepath : type
        Description of parameter `savepath`.
    outputfn : type
        Description of parameter `outputfn`.
    model : type
        Description of parameter `model`.
    X_train : type
        Description of parameter `X_train`.
    y_train : type
        Description of parameter `y_train`.
    X_test : type
        Description of parameter `X_test`.
    y_test : type
        Description of parameter `y_test`.

    Returns
    -------
    type
        Description of returned object.

    """
    textPath = savepath + '/text/'
    if not params['GP']:
        textfile = open(textPath + "/" + outputfn + "_%s.txt" % ts, "wt")
    else:
        textfile = open(textPath + "/" + outputfn + "_%s_%s.txt"%(ts, params['band_stack']), "wt")
    for key in params.keys():
        textfile.write("{} = {}\n".format(key, params[key]))
    textfile.write("Training:\n")
    with redirect_stdout(textfile):
        # fit the data!
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params['batch_size'], epochs=params['nepochs'], verbose=2)
        model.summary()
    textfile.close()
