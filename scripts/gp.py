import jaxopt
import numpy as np
import pandas as pd
import tinygp
import jax
import jax.numpy as jnp
from io import StringIO
import matplotlib.pyplot as plt
from plotting import *
import multiprocessing
from helper_functions import *
import pickle
from jax.config import config

config.update("jax_enable_x64", True)

class Multiband(tinygp.kernels.Kernel):
    """The multi-band model for the Gaussian process.

    Parameters
    ----------
    time_kernel : tinygp Kernel
        A kernel describing the correlations between observations in time.
    diagonal : type
        Description of parameter `diagonal`.
    off_diagonal : type
        Description of parameter `off_diagonal`.

    Attributes
    ----------
    band_kernel : type
        Description of attribute `band_kernel`.
    time_kernel

    """
    def __init__(self, time_kernel, diagonal, off_diagonal):
        ndim = diagonal.size
        if off_diagonal.size != ((ndim - 1) * ndim) // 2:
            raise ValueError(
                "Dimension mismatch: expected "
                f"(ndim-1)*ndim/2 = {((ndim - 1) * ndim) // 2} elements in "
                f"'off_diagonal'; got {off_diagonal.size}"
            )

        factor = jnp.zeros((ndim, ndim))
        factor = factor.at[jnp.diag_indices(ndim)].add(diagonal)
        factor = factor.at[jnp.tril_indices(ndim, -1)].add(off_diagonal)
        self.band_kernel = factor @ factor.T
        self.time_kernel = time_kernel

    def evaluate(self, X1, X2):
        t1, b1 = X1
        t2, b2 = X2
        return self.band_kernel[b1, b2] * self.time_kernel.evaluate(t1, t2)

def sngp_slice(slice_dict, params, plotpath='./', bands='XY', GPmethod='100Timestep', fn='GP'):
    """The light curve gaussian process interpolation function.

    Parameters
    ----------
    slice_dict : dictionary
        The dictionary containing the light curve segments (before interpolation).
    params : dictionary
        The full dictionary of parameter values (including all parameters for the GP kernel).
    plotpath : str
        Path to the dictionary where data will be saved.
    bands : str
        The bandpasses over which to compute interpolated light curves.
    GPmethod : str
        The GP method to use (`100Timestep` to interpolate 100 points from start to finish of each segment
         or `0.2Day` to interpolate N points with 0.2-day spacing from start to finish of each segment)
    fn : str
        The filename to save dictionaries of interpolated light curves.

    Returns
    -------
    Dictionary
        The dictionary of interpolated values.

    """

    num_bands = len(bands)
    GP_dict = {}

    GP_dict['GP_T'] = []
    GP_dict['GP_Flux'] =  []
    GP_dict['GP_Flux_Err'] =  []
    GP_dict['GP_Filter'] =  []

    nn_param_names = ['mean', 'log_scale', 'log_diagonal', 'off_diagonal', 'log_jitter']
    params_nn = {key: value for key, value in params.items() if key in nn_param_names}

    band_dict = {}
    ctr = 0
    for band in bands:
        band_dict[band] = ctr
        ctr += 1

    colset = sns.color_palette()
    cols_bands = [colset[4], colset[0], colset[2], colset[8], colset[1], colset[3]]
    i = 0
    paramDF_list = []
    t = np.array(slice_dict["T"])
    f = np.array(slice_dict["Flux"])
    f[f<0.] = 0. #getting rid of negative flux

    band_idx = np.array([band_dict[x] for x in np.array(slice_dict['Filter']) if x in bands])

    #the magnitude-like array for the sake of the conversion
    y = np.log(f + 1)
    yerr = np.array(slice_dict["Flux_Err"]) / np.array(slice_dict["Flux"])
    tmin = np.nanmin(t)
    tmax = np.nanmax(t)
    if np.nanmin(t) < 0.:
        tmin = 0.
    flt = np.array(slice_dict['Filter'])

    if GPmethod == '100Timestep':
        t_test = np.linspace(tmin, tmax, 100)
    elif GPmethod == '0.2Day':
        t_test = np.arange(tmin, tmax, 0.2)

    def build_gp(params):
        time_kernel = tinygp.kernels.Matern32(jnp.exp(params["log_scale"]))
        kernel = Multiband(time_kernel, jnp.exp(params["log_diagonal"]), params["off_diagonal"])
        diag = yerr ** 2 + jnp.exp(2 * params["log_jitter"][X[1]])
        return tinygp.GaussianProcess(kernel, X, diag=diag, mean=lambda x: params["mean"][x[1]])

    @jax.jit
    def loss(params):
        return -build_gp(params).log_probability(y)
    X = (t, band_idx)

    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params_nn)
    gp = build_gp(soln.params)

    paramDict = soln.params.copy()
    for key, val in paramDict.items():
        val = np.array(val)
        if val.size > 1:
            paramDict[key] = [val]

    df_t = []
    df_flux = []
    df_flux_err = []
    df_filt = []

    ymax = 0.
    ymin = 1.e8
    for n in np.arange(len(bands)):
        m = np.array(flt) == bands[n]
        mu, var = gp.predict(y, X_test=(t_test, np.full_like(t_test, n, dtype=int)), return_var=True)
        std = np.sqrt(var)
        try:
            tempymax =np.nanmax(np.exp(y[m])-1)
            tempymin = np.nanmin(np.exp(y[m])-1)
            if tempymax > ymax:
                ymax = tempymax
            if tempymin < ymin:
                ymin = tempymin
        except:
            print("No values for %s band!\n"%bands[n])

        gp_f =np.exp(mu)-1
        gp_f_err = np.abs(np.exp(mu + std) - np.exp(mu - std))/2

        GP_dict['GP_T'].append(t_test)
        GP_dict['GP_Flux'].append(gp_f)
        GP_dict['GP_Filter'].append([bands[n]]*len(gp_f))
        GP_dict['GP_Flux_Err'].append(gp_f_err)

    for key, val in GP_dict.items():
        GP_dict[key] = np.concatenate(GP_dict[key])
    return GP_dict

def createStackedDict(df):
    """A function to build the dictionary of interpolated segments from
    the original dataframe. This function is passed to multiprocessing's Pool
    to interpolate light curve segments in parallel.

    Parameters
    ----------
    df : Pandas DataFrame
        The full dataset containing the sliced light curve segments to interpolate.

    """
    ts = int(time.time())
    GP_dict = {}
    for idx, row in df.iterrows():
        try:
            t = row['T']
            flux = row['Flux']
            filter = row['Filter']
            fluxerr = row['Flux_Err']

            k=0

            CID = row['CID']

            slice_dict = {'T':t, 'Flux':flux, 'Flux_Err':fluxerr, 'Filter':filter}

            final_dict = sngp_slice(slice_dict, params, plotpath='./', bands='XY', GPmethod='100Timestep', fn='GP')
            matrix = [final_dict['GP_T'][final_dict['GP_Filter'] == 'X']]

            GPfilt = final_dict['GP_Filter']
            for band in 'XY':
                matrix = np.vstack([matrix, final_dict['GP_Flux'][GPfilt == band], final_dict['GP_Flux_Err'][GPfilt == band]])
            GP_dict["%s_%i" %(str(CID), i)] = matrix
            if len(temp_t) == 0:
                continue
        except:
            print("Error in SN %i"%int(CID))
    with open('ZTFSims_GP_%i_%s_%i.pkl'%(CID, params['GPMethod'], ts), 'wb') as f:
        pickle.dump(GP_dict, f)

def gen_gp(df):
    """Wrapper function to generate the gaussian process-interpolated light curve segments.

    Parameters
    ----------
    df : Pandas DataFrame
        The full dataset containing the sliced light curve segments to interpolate.
    """

    df = correct_time_dilation(df)
    df = correct_extinction_flux(df)

    t_init = time.time()
    print("Time to parallelize!")
    num_cores = multiprocessing.cpu_count()-4 #leave one free to not freeze machine
    print("Number of cores is", num_cores)
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    print("Number of LCs in each dictionary: %i" %len(df_split[0]))
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(createStackedDict, df_split))
    pool.close()
    pool.join()

    t_final = time.time()
    print(t_final - t_init)
