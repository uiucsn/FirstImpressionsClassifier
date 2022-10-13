import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from extinction import ccm89, apply, remove
import extinction
from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import FlatLambdaCDM


#The limiting magnitude of your survey
MAG_LIM = 31.0
ZPT = 30.0

bands = 'grizXY'
wv_dict =  {'g':5460, 'r':6800, 'i':7450, 'z':8700, 'X':4767, 'Y': 6215}
wvs = np.asarray(list(wv_dict.values()))

def segment_df(df, outfn='./DF_Sliced.json'):
    """Generates and saves a dataframe of light curve segments. Host galaxy values are copied
    across rows for segments of the same light curve.

    Parameters
    ----------
    df : Pandas DataFrame
        The original dataframe of photometry and host galaxy properties.
    outfn : str
        The name for the saved segmented dataframe.

    Returns
    -------
    Pandas Dataframe
        The segmented dataframe.
    """
    #slice up the dataset into observation segments
    df = []
    for idx, row in df.iterrows():
        mag = np.array(row['MAG'])
        magerr = np.array(row['MAGERR'])
        mjd = np.array(row['MJD'])
        flux = np.array(row['Flux'])
        filter = np.array(row['Filter'])
        flux_err = np.array(row['Flux_Err'])
        T = np.array(row['T'])

        for i in np.arange(len(T)):
            slicedrow = df.iloc[[idx]]
            slicedrow['CID'] = slicedrow['CID'] + "_pt" + str(i)
            slicedrow['MAG'] = [mag[0:i+1]]
            slicedrow['MAGERR'] = [magerr[0:i+1]]
            slicedrow['MJD'] = [mjd[0:i+1]]
            slicedrow['Flux'] = [flux[0:i+1]]
            slicedrow['Filter'] = [filter[0:i+1]]
            slicedrow['Flux_Err'] = [flux_err[0:i+1]]
            slicedrow['T'] = [T[0:i+1]]
            df_slicedList.append(slicedrow)
    df_sliced = pd.concat(df_slicedList)
    df_sliced.reset_index(drop=True, inplace=True)
    df_sliced.to_json(outfn)
    return df_sliced

def correct_time_dilation(df):
    """Function to correct phase information for time-dilation.

    Parameters
    ----------
    df  : Pandas DataFrame
        The dataframe containing the photometry of all events.

    Returns
    -------
    Pandas DataFrame
        The same dataframe with time-diliation-corrected phases.

    """
    zName = list(set(['GALZPHOT', 'ZCMB', 'HOSTGAL_PHOTOZ']).intersection(df.columns.values))[0]
    for idx, row in df.iterrows():
        row['T'] = row['T'] / (1.+row[zName])
    return df

def correct_extinction_flux(df):
    """Corrects photometry for milky way extinction (requires MWEBV in the pandas dataframe!).

    Parameters
    ----------
    df  : Pandas DataFrame
        The dataframe containing the photometry of all events.
    wvs : array-like
        Description of parameter `wvs`.

    Returns
    -------
    type
        Description of returned object.

    """
    for idx, row in df.iterrows():
        tempFlux = np.array(row.Flux)
        filt = row.Filter
        for band in np.array(list(wv_dict.keys())):
            ii = filt == band
            tempFlux[ii] = remove(ccm89(np.array([float(wv_dict[band])]), 3.1*row.MWEBV, 3.1), tempFlux[ii])
        df.at[idx, 'Flux'] = tempFlux
    return df

def calc_gal_lum(df, params, system='PS1'):
    """Converts apparent galaxy magnitudes to normalized luminosity.

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe containing the photometry of all events.
    params : dictionary
        The parameters of the run (needed for the host-galaxy passbands)
    system : str
        The filter system from which to calculate absolute magnitudes (used to calculate zeropoints).

    Returns
    -------
    Pandas DataFrame
        The dataframe containing the normalized host-galaxy luminosity.

    """
    if params['verbose']:
        print("Calculating luminosity...")
    if system=='PS1':
        ZPTs = {'g':24.583, 'r':24.783, 'i':24.635, 'z':24.278}
    elif (system == 'LSST') | (system == 'VRO'):
        ZPTs = {'g':31.50756069364162, 'r':31.527203311867524, 'i':31.325204545454543, 'z':31.001521410579343}

    zName = list(set(['GALZPHOT', 'ZCMB']).intersection(df.columns.values))[0]

    # returns dist in Mpc
    df['DIST'] = cosmo.luminosity_distance([df[zName]]).value[0]

    for band in params['hostBands']:
        df['HOSTGAL_LUM_%s'%band]= np.nan

        # first, convert to flux
        df['HOSTGAL_FLUX_%s' % band] = 10**((ZPTs[band] - df['HOSTGAL_MAG_%s' % band])/2.5)

        #divided by some normalization constant to have all values around 1
        scaleFac = 4*np.pi*(df['DIST']*1.e6)**2/1.e23
        lum = scaleFac*df['HOSTGAL_FLUX_%s' % band]
        df['HOSTGAL_LUM_%s'%band] = lum
    return df

def calc_lum(df, params, err_fill=0.0):
    """Converts the light curves to normalized luminosity.

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe containing the photometry of all events.
    params : dictionary
        The parameters of the run (needed for the `verbose` option).
    err_fill : float
        The value to set for the luminosity error (if the calculation fails).

    Returns
    -------
    Pandas DataFrame
        The dataframe with normalized luminosity for each SN.

    """
    if params['verbose']:
        print("Calculating luminosity...")

    df['GP_Lum'] = [[np.nan]]*len(df)
    df['GP_Lum_Err'] =  [[np.nan]]*len(df)

    zName = list(set(['GALZPHOT', 'ZCMB']).intersection(df.columns.values))[0]

    for idx, row in df.iterrows():
        dist = cosmo.luminosity_distance([row[zName]]).value[0]  # returns dist in Mpc
        scaleFac = 4*np.pi*(dist*1.e6)**2/1.e23 #divided by some normalization constant to match RAPID
        lum = scaleFac*row['GP_Flux']
        lum_err = scaleFac*row['GP_Flux_Err']

        df.at[idx, 'GP_Lum'] = lum
        df.at[idx, 'GP_Lum_Err'] = lum_err
    return df

def stackInputs(df, params):
    """Translates a pandas dataframe into a dictionary of numpy matrices for photometry of all events.

    Parameters
    ----------
    df    : Pandas DataFrame
        The dataframe containing the photometry of all events.
    params : dict
        Dictionary of all run params.

    Returns
    -------
    LCs
        The dictionary of all stacked photometry.

    """
    LCs = {}
    bands = params['bands'] #use all bands if we have gp-interpolation for them!
    for idx, row in df.iterrows():
        SN = np.array(row.CID)
        Time = np.array(row.GP_T)
        phot = np.array(row.GP_Lum)
        phot_Err = np.array(row.GP_Lum_Err)
        Filt = np.array(row.GP_Filter)

        bandTimes = Time[Filt==bands[0]]
        matrix = [bandTimes]

        for i in np.arange(len(bands)):
            band = bands[i]
            bandPhot =  phot[Filt==band]
            bandPhotErr = phot_Err[Filt==band]
            matrix = np.vstack([matrix, bandPhot, bandPhotErr])
        LCs[row.CID] = matrix
    return LCs

def encode_classes(df):
    """Encodes the output classes as integers and returns a
    dictionary of the encodings.

    Parameters
    ----------
    df    : Pandas DataFrame
        The dataframe containing the photometry of all events.

    Returns
    -------
    Pandas DataFrame
        The same dataframe with encoded column Type_ID.
    Pandas dict
        Dictionary of encoded classes.
    """
    df['Type_ID'] = df['Type'].astype('category').cat.codes

    #some clunky trickery to get the mapping from classes to values
    encoding_dict = df[['Type', 'Type_ID']].drop_duplicates(subset=['Type', 'Type_ID']).sort_values(by='Type_ID').reset_index(drop=True)['Type'].to_dict()
    return df, encoding_dict

def addHostFeatures(params, stackedInputs, fullDF, textfile):
    """Stacks the host galaxy properties onto the dictionary of transient phoeomtry.
    The features used can be set in the `params` dictionary.

    Parameters
    ----------
    params : Dictionary
        The full dictionary of parameter values (including the host galaxy features to add).
    stackedInputs : Dictionary
        Stacked representations of the photometry of each SN (contains phase, luminosity, and luminosity error in each band).
    fullDF : Pandas DataFrame
        The full dataframe of data (train and test sets).
    textfile : str
        The file to write out updates as the host galaxy features are added.

    Returns
    -------
    dictionary
        The data dictionaries with both transient photometry and host-galaxy properties.

    """

    #temp key for getting length of the items in the dictionary
    key_temp = list(stackedInputs.keys())[0]

    if params['z']:
        zName = list(set(['GALZPHOT', 'ZCMB']).intersection(fullDF.columns.values))[0]
        params['Nfeatures'] += 1
        if params['verbose']:
            print("Adding in z...", file=textfile)
            if zName == 'ZCMB':
                print("Warning! It looks like you're using true redshift. Careful, this could yield overly optimistic results...", file=textfile)
        NZ = np.shape(stackedInputs[key_temp])[-1]
        for key_temp, val in stackedInputs.items():
            tempZ = fullDF.loc[fullDF['CID'] == key_temp, zName].values[0]
            stackedInputs[key_temp] = np.vstack([stackedInputs[key_temp], [tempZ]*NZ])
    if params['MWEBV']:
        params['Nfeatures'] += 1
        if params['verbose']:
            print("Adding in MWEBV...", file=textfile)
        NMW = np.shape(stackedInputs[key_temp])[-1]
        for key_temp, val in stackedInputs.items():
            tempMWEBV = fullDF.loc[fullDF['CID'] == key_temp, 'MWEBV'].values[0]
            stackedInputs[key_temp] = np.vstack([stackedInputs[key_temp], [tempMWEBV]*NMW])
    if params['hostPhot']:
        params['Nfeatures'] += 4
        if params['verbose']:
            print("Adding in host galaxy photometry...", file=textfile)
        Nband = np.shape(stackedInputs[key_temp])[-1]
        for band in 'griz':
            for key_temp, val in stackedInputs.items():
                tempphot = fullDF.loc[fullDF['CID'] == key_temp, 'HOSTGAL_MAG_%s'%band].values[0]
                stackedInputs[key_temp] = np.vstack([stackedInputs[key_temp], [tempphot]*Nband])
    return stackedInputs

def build_rnn_arrays(params, df_train, df_test):
    """

    Parameters
    ----------
    params : dictionary
        The full dictionary of parameter values (including all parameters for the GP kernel).
    df_train : Pandas Dataframe
        The training set (transient and host galaxy data).
    df_test : Pandas Dataframe
        The test set (transient and host galaxy data).

    Returns
    -------
    array-like
        The feature array of the training set.
    array-like
        The target vector for the classes of the training set.
    array-like
        The feature array of the test set.
    array-like
        The target vector for the classes of the test set.

    """
    df_train, encoding_dict = encode_classes(df_train)
    df_test, encoding_dict = encode_classes(df_test)

    fullDF = pd.concat([df_train, df_test], ignore_index=True)

    snclasses = np.unique(list(encoding_dict.values()))
    params['Nclass'] = len(snclasses)

    stackedInputs_test = stackInputs(df_test, params)
    stackedInputs_train = stackInputs(df_train, params)

    print("Training set is %i objects. Among the training set:" % len(df_train), file=textfile)

    for i in np.arange(N_class):
        tempDF = df_train[df_train['Type'] == snclasses[i]]
        print("%s: %i" % (snclasses[i], len(tempDF)), file=textfile)

    print("Test set is %i objects. Among the testing set:" % len(df_test), file=textfile)

    for i in np.arange(N_class):
        tempDF = df_test[df_test['Type'] == snclasses[i]]
        print("%s: %i" % (snclasses[i], len(tempDF)), file=textfile)

    stackedInputs_train = addHostFeatures(stackedInputs_train)
    stackedInputs_test = addHostFeatures(stackedInputs_test)

    #set up the train and test sets
    X_train = np.swapaxes(list(stackedInputs_train.values()), 1, 2)
    X_test = np.swapaxes(list(stackedInputs_test.values()), 1, 2)

    y_train = df_train['Type_ID'].values
    y_test = df_test['Type_ID'].values
    return X_train, y_train, X_test, y_test

def pad_dicts(params, train_dict, test_dict):
    """Adds padding zeros to the tail end of each dictionary item of interpolated light curves

    Parameters
    ----------
    params : dictionary
        The full dictionary of parameter values (including the number of features used in the model).
    train_dict : dictionary
        The full dictionary of interpolated light curves from the training set.
    test_dict : dictionary
        The full dictionary of interpolated light curves from the test set.

    Returns
    -------
    dictionary
        The padded dictionary of interpolated light curves from the training set.
    dictionary
        The padded dictionary of interpolated light curves from the test set.
    """
    #pad to the ends to make sure they're all the same length!
    longestLen = 0
    for key, val in train_dict.items():
        if (np.shape(val)[-1] > longestLen):
            longestLen = np.shape(val)[-1]
    for key, val in test_dict.items():
        if (np.shape(val)[-1] > longestLen):
            longestLen = np.shape(val)[-1]
    longestLen

    train_dict_padded = {}
    test_dict_padded = {}

    for key, val in train_dict_padded.items():
        pad_matrix = np.zeros((params['Nfeatures'], longestLen - np.shape(val)[-1]))
        train_dict_padded[key] = np.hstack([val, pad_matrix])
    for key, val in test_dict_padded.items():
        pad_matrix = np.zeros((params['Nfeatures'], longestLen - np.shape(val)[-1]))
        test_dict_padded[key] = np.hstack([val, pad_matrix])
    return train_dict_padded, test_dict_padded

def jitter_lcs(df, headrand=0.25):
    """Randomly truncates the initial points of each light
    curve to reduce the reliance on the survey trigger.

    Parameters
    ----------
    df : Pandas DataFrame
        The dataset containing the photometry for all events.
    headrand : float
        The standard deviation of a normal distribution to draw from to
        calculate the new trigger time (relative to the old one) in days.

    Returns
    -------
    Pandas DataFrame
        The dataframe with jittered light curve segments.

    """
    for idx, row in df.iterrows():
        mint = (row['MJD_TRIGGER'])
        tmin = np.random.normal(0, headrand)

        flux = np.array(row['Flux'])
        fluxerr = np.array(row['Flux_Err'])
        filt = np.array(row['Filter'])
        t = np.array(row['T'])
        photflag = np.array(row['PHOTFLAG'])
        mag = np.array(row['MAG'])
        snr = np.array(row['SNR'])
        magerr = np.array(row['MAGERR'])
        mjd = np.array(row['MJD'])

        bool = mjd >= tmin
        if np.nansum(bool) < 2:
            continue
        else:
            df.at[idx, 'Flux'] = flux[bool]
            df.at[idx, 'Flux_Err'] = fluxerr[bool]
            df.at[idx, 'MAG'] = mag[bool]
            df.at[idx, 'MAGERR'] = magerr[bool]
            df.at[idx, 'T'] = mjd[bool] - tmin
            df.at[idx, 'Filter'] = filt[bool]
            df.at[idx, 'PHOTFLAG'] = photflag[bool]
            df.at[idx, 'SNR'] = snr[bool]
            df.at[idx, 'MJD'] = mjd[bool]
    return df

def add_segments_to_df(params, DF, GPdict):
    """Adds interpolated segments from dictionary back to full DataFrame.

    Parameters
    ----------
    params : dictionary
        The full dictionary of parameter values (including the bands of SN photometry).
    DF : Pandas DataFrame
        The segmented dataframe containing host galaxy and transient photometry.
        raw photometry.
    GPdict : dictionary
        The interpolated light curve segments.

    Returns
    -------
    Pandas DataFrame
        The full dataframe, with interpolated segments.

    """
    DF = DF[DF['CID'].isin(GPdict.keys())]

    toDrop = set(list(GPdict.keys())) - set(DF['CID'])
    for transient in toDrop:
           remove_key = GPdict.pop(transient, None)

    DF['GP_T'] = DF['T']
    DF['GP_Flux'] = DF['Flux']
    DF['GP_Flux_Err'] = DF['Flux_Err']
    DF['GP_Filter'] = DF['Filter']
    for key, val in GPdict.items():
        idx = DF.loc[DF['CID'] == key].index[0]
        tm =  np.tile(val[0], len(params['bands']))
        DF.at[idx, 'GP_T'] = tm
        DF.at[idx, 'GP_Flux'] = []
        DF.at[idx, 'GP_Flux_Err'] = []
        DF.at[idx, 'GP_Filter'] = []
        flux = []
        flux_err = []
        filters = []

        for i in np.arange(len(params['bands'])):
            j = 2*i+1
            band = params['bands'][i]
            filters.append([band]*np.shape(val)[-1])
            flux.append(val[j])
            flux_err.append(val[j+1])
        DF.at[idx, 'GP_Flux'] = np.concatenate(flux)
        DF.at[idx, 'GP_Flux_Err'] = np.concatenate(flux_err)
        DF.at[idx, 'GP_Filter'] = np.concatenate(filters)

    DF = calc_lum(DF, params=params, err_fill=1.0)
    return DF
