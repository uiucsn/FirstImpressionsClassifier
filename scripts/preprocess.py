import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import extinction
from astropy.cosmology import Planck13 as cosmo

#The limiting magnitude of your survey
MAG_LIM = 33.0
ZPT = 30.0
wvs = np.asarray([3600, 4760, 6215, 7545, 8700, 10150])
bands = 'ugrizY'

def shift_lc(df):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.

    Returns
    -------
    type
        Description of returned object.

    """
    df['T'] = df['MJD'] - df['MJD_TRIGGER']
    return df

def cut_lc(df, min=-30, max=150):
    for idx, row in df.iterrows():
        Times = row['T']
        Flux =  row['Flux']
        Flux_Err = row['Flux_Err']
        Filter = row['Filter']
        MJD = row['MJD']

        #truncate
        ii = (Times > min) & (Times < max)
        Flux = Flux[ii]
        Flux_Err = Flux_Err[ii]
        Times = Times[ii]
        Filter = Filter[ii]
        MJD = MJD[ii]

        df.at[idx, 'T'] = Times
        df.at[idx, 'Filter'] = Filter
        df.at[idx, 'MJD'] = MJD
        df.at[idx, 'Flux'] = Flux
        df.at[idx, 'Flux_Err'] = Flux_Err
    return df

def correct_time_dilation(df):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.

    Returns
    -------
    type
        Description of returned object.

    """
    for idx, row in df.iterrows():
        row['T'] = row['T'] / (1.+row.ZCMB)
    return df

def correct_extinction(df, wvs):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.
    wvs : type
        Description of parameter `wvs`.

    Returns
    -------
    type
        Description of returned object.

    """
    for idx, row in df.iterrows():
        alams = extinction.fm07(wvs, row.MWEBV)
        tempMag = np.array(row.Mag)
        for i, alam in enumerate(alams):
            if bands[i] in row.Filter:
                ii = np.array(row.Filter)[0] == bands[i]
                tempMag[ii] -= alam
        df.at[idx, 'Mag'] = tempMag
    return df

def read_in_LC_data(metafile='./metafile.txt', LC_path='./', format='SNANA'):
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
    return LCs.merge(metatable)

def calc_abs_mags(df, err_fill=1.0):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.

    Returns
    -------
    type
        Description of returned object.
    """
    df['Mag'] = [[np.nan]]*len(df)
    df['Mag_Err'] =  [[np.nan]]*len(df)
    df['Abs_Lim_Mag'] = np.nan

    for idx, row in df.iterrows():
        k_correction = 2.5 * np.log10(1.+row.ZCMB)
        dist = cosmo.luminosity_distance([row.ZCMB]).value[0]  # returns dist in Mpc

        abs_mags = -2.5 * np.log10(row.Flux) + ZPT - 5. * \
            np.log10(dist*1e6/10.0) + k_correction

        # Sketchy way to calculate error - update later
        abs_mags_plus_err = -2.5 * np.log10(row.Flux + row.Flux_Err) + ZPT - 5. * \
            np.log10(dist*1e6/10.0) + k_correction
        abs_mags_err = np.abs(abs_mags_plus_err - abs_mags)
        abs_lim_mag = MAG_LIM - 5.0 * np.log10(dist * 1e6 / 10.0) + \
                            k_correction

        abs_mags_err[abs_mags != abs_mags] = err_fill
        abs_mags[abs_mags != abs_mags] = abs_lim_mag

        df.at[idx, 'Mag'] = abs_mags
        df.at[idx, 'Mag_Err'] = abs_mags_err
        df.at[idx, 'Abs_Lim_Mag'] = abs_lim_mag
    return df

#def getGPLCs(df):
def stackInputs(df, bands='ugrizY', pad_cadence=2.0):
    """Some basic description

    Parameters
    ----------
    df : type
        Description of parameter `df`.
    bands : type
        Description of parameter `bands`.

    Returns
    -------
    type
        Description of returned object.

    """
    LCs = {}
    #get max length of a matrix
    maxLen = np.nanmax([len(x) for x in df['MJD'].values])
    for idx, row in df.iterrows():
        SN = row.CID
        Time = row['T']
        Mag = row.Mag
        Mag_Err = row.Mag_Err
        Filt = row.Filter
        for band in bands:
            matrix = np.zeros((maxLen, 3))
            if np.nansum(Filt==band) == 0:
                continue
            bandTimes = Time[Filt==band]
            bandMags =  Mag[Filt==band]
            bandErrs = Mag_Err[Filt==band]

            #get GP LCs later -- for now, just pad
            #pad fill
            padLen = maxLen - len(bandMags)
            abs_mag_lim = df.at[idx, 'Abs_Lim_Mag'].astype(np.float64)
            padR = int(padLen/2)
            padF = padR
            if padLen%2 == 1:
                #pad more on the forward end than the back end
                padF += 1

            padArr_R = [abs_mag_lim]*padR
            padErr_R = [1.0]*padR

            padArr_F = [abs_mag_lim]*padF
            padErr_F = [1.0]*padF
            timePad_R = -np.arange(0,padR)*pad_cadence-pad_cadence + np.nanmin(bandTimes)
            np.flip(timePad_R)
            timePad_F = np.arange(0,padF)*pad_cadence + pad_cadence + np.nanmax(bandTimes)

            #combine
            stackTimes = np.concatenate([timePad_R, bandTimes, timePad_F])
            stackMags = np.concatenate([padArr_R, bandMags, padArr_F])
            stackErrs = np.concatenate([padErr_R, bandErrs, padErr_F])

            matrix = np.vstack([stackTimes, stackMags, stackErrs])
            LCs[row.CID] = matrix
    return LCs


#def getGPLCs(df):
def stackGPInputs(df, bands='ugrizY'):
    """Some basic description

    Parameters
    ----------
    df : type
        Description of parameter `df`.
    bands : type
        Description of parameter `bands`.

    Returns
    -------
    type
        Description of returned object.

    """
    LCs = {}
    #get max length of a matrix
    for idx, row in df.iterrows():
        SN = row.CID
        Time = row['GP_T']
        Flux = row['GP_Flux']
        Flux_Err = row['GP_Flux_Err']
        Filt = row['GP_Filter']
        #in the GP model, we're at the same time for everything
        Time = Time[Filt == 'u'] #pick any band, doesn't matter
        maxLen = len(Time)
        for band in bands:
            matrix = np.zeros((maxLen, len(bands)*2+1)) #ugrizY Flux, ugrizY err, time
            bandFlux =  Flux[Filt==band]
            bandErrs = Flux_Err[Filt==band]

            #get GP LCs
            if bands == 'u':
                matrix = np.vstack([stackTimes, bandFlux, bandErrs])
            else:
                matrix = np.vstack([matrix, bandFlux, bandErrs])
            LCs[row.CID] = matrix
    return LCs

def encode_classes(df, verbose=True):
    df['Type_ID'] = df['Type'].astype('category').cat.codes
    #some clunky trickery to get the mapping from classes to values
    encoding_dict = df[['Type', 'Type_ID']].drop_duplicates(subset=['Type', 'Type_ID']).sort_values(by='Type_ID').reset_index(drop=True)['Type'].to_dict()
    return df, encoding_dict
