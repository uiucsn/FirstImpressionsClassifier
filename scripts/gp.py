import jaxopt
import numpy as np
import pandas as pd
import tinygp
import jax
import jax.numpy as jnp
from io import StringIO
import matplotlib.pyplot as plt
from plotting import *
from jax.config import config
config.update("jax_enable_x64", True)
bands = 'ugrizY'
N_bands = len(bands)

class Multiband(tinygp.kernels.Kernel):
    """Short summary.

    Parameters
    ----------
    time_kernel : type
        Description of parameter `time_kernel`.
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


# Right now the GP DOES NOT have a pad. What it SHOULD have (I think) is the following:
# Define newTime = np.linspace(-30, 150, 100)
# Initialize gpF_err = np.ones(100)
#            gpF = np.zeros(100)
# Go band-by-band:
#        For band i, get t0 as np.nanmax(-30, min(bandT))
#                        tf as np.nanmin(150, max(bandT))
#        Define gpTime as newTime truncated between t0 and tf, get index i of first value of gpTime in newTime
#        Run GP and evaluate at gpTime values
#        Fill gpF and gpF_err starting at i for all values in gpTime
# DONE if all goes well, you should have 100 points for every band between -30 and 150 days
# evenly spaced for all observations and without needing to extrapolate.
# then you don't even need to pass in the time parameter for every LC, reducing the dimensionality!
#def gp_withPad(df, savepath='./',plotpath='./', bands='ugrizY', Ntstp=100, ts='0000000', fn='GPSet'):
#will this cause issues on the front end? Potentially. Let's find out.
def gp_withPad(df, savepath='./',plotpath='./', bands='ugrizY', Ntstp=100, ts='0000000', fn='GPSet'):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.
    savepath : type
        Description of parameter `savepath`.
    plotpath : type
        Description of parameter `plotpath`.
    bands : type
        Description of parameter `bands`.
    ts : type
        Description of parameter `ts`.
    fn : type
        Description of parameter `fn`.

    Returns
    -------
    type
        Description of returned object.

    """
    #num_bands = len(np.unique(band_idx))
    num_bands = len(bands)
    GP_dict = {}

    #only interpolate from tmin to tmax, and then pad the ends in order to get to 100 points!
    for idx, row in df.iterrows():
        t = np.array(row["T"])
        f = np.array(row["Flux"])
        f[f<0.] = 0. #getting rid of negative flux

        #the magnitude-like array for the sake of the conversion
        y = np.log(f + 1)
        yerr = np.array(row["Flux_Err"]) / np.array(row["Flux"])
        t_test = np.linspace(np.nanmin(t), np.nanmax(t), Ntstp) #only go from tmin to tmax
        band = row["Filter"]
        band_idx = pd.Series(row['Filter']).astype('category').cat.codes.values

        #padL = Ntstp - len(t_test) #how many observations to we need to tack onto the end?
        ##generate spacing
        #padT = np.arange(padL)+1 #one-day spacing tacked onto the end of the interpolated sequence
        #df_T = np.concatenate([t_test, padT])
        #matrix = [df_T]
        #we shouldn't need to pad -- figure this out later
        padL = 0 # don't do any padding for now
        matrix = [t_test]

        def build_gp(params):
            time_kernel = tinygp.kernels.Matern32(jnp.exp(params["log_scale"]))
            kernel = Multiband(time_kernel, jnp.exp(params["log_diagonal"]), params["off_diagonal"])
            diag = yerr ** 2 + jnp.exp(2 * params["log_jitter"][X[1]])
            return tinygp.GaussianProcess(kernel, X, diag=diag, mean=lambda x: params["mean"][x[1]])

        #the GP parameters
        params = {
            "mean": np.zeros(num_bands),
            "log_scale": np.log(100.0),
            "log_diagonal": np.zeros(num_bands),
            "off_diagonal": np.zeros(((num_bands - 1) * num_bands) // 2),
            "log_jitter": np.zeros(num_bands),
        }
        @jax.jit
        def loss(params):
            return -build_gp(params).condition(y)

        X = (t, band_idx)

        solver = jaxopt.ScipyMinimize(fun=loss)
        soln = solver.run(params)
        gp = build_gp(soln.params)

        df_t = []
        df_flux = []
        df_flux_err = []
        df_filt = []

        if idx%50 == 0:
            print("Plotting %i...\n"%idx)
            plt.figure(figsize=(10,7))
        for n in np.unique(band_idx):
            m = band_idx == n
            plt.errorbar(t[m], np.exp(y[m])-1,yerr=row['Flux_Err'][m], fmt="o", color=f"C{n}")
            mu, var = gp.predict(y, X_test=(t_test, np.full_like(t_test, n, dtype=int)), return_var=True)
            std = np.sqrt(var)
            if idx%50 == 0:
                plt.plot(t_test, np.exp(mu)-1, '.-', ms=2, color=f"C{n}")
                plt.fill_between(t_test,np.exp(mu - std)-1, np.exp(mu + std)+1, color=f"C{n}", alpha=0.3, label=bands[n])

            #going in order of band here--don't forget it! (ugrizY)
            #now pad the end
            padF = np.zeros(padL) #one-day spacing tacked onto the end of the interpolated sequence
            padFerr = np.ones(padL)

            gp_f = np.concatenate([np.exp(mu)-1, padF])
            gp_f_err = np.concatenate([std, padFerr])

            matrix.append(gp_f)
            matrix.append(gp_f_err)

            df_t.append(t[m])
            df_flux.append(gp_f)
            df_flux_err.append(gp_f)
            df_filt.append([bands[n]]*len(t[m]))

        if idx%50 == 0:
            plotmin = np.nanmin([t_test[0], -30])
            plotmax = np.nanmax([t_test[-1], 150])
            plt.xlim((plotmin, plotmax))
            plt.xlabel("Phase from Trigger (Days)")
            plt.ylabel("Flux")
            plt.legend()
            plt.savefig(plotpath + "/GP_%i.png"%row.CID,dpi=200, bbox_inches='tight')

        stacked = np.vstack(matrix)
        GP_dict[row.CID] = stacked

        # overwrite the original data (not a great solution, but potentially better
        # for all the other functions that will use these column names)
        df.at[idx, 'T'] = np.concatenate(df_t)
        df.at[idx, 'Filter'] = np.concatenate(df_filt)
        df.at[idx, 'Flux'] = np.concatenate(df_flux)
        df.at[idx, 'Flux_Err'] = np.concatenate(df_flux_err)

    #save the dictionary separately just to have them
    with open(savepath + '/%s_%i.pkl'%(fn, ts), 'wb') as f:
        pickle.dump(GP_dict, f)

    return df

def getGPLCs(df, savepath='./',plotpath='./', bands='ugrizY', ts='0000000', fn='GPSet'):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.
    savepath : type
        Description of parameter `savepath`.
    plotpath : type
        Description of parameter `plotpath`.
    bands : type
        Description of parameter `bands`.
    ts : type
        Description of parameter `ts`.
    fn : type
        Description of parameter `fn`.

    Returns
    -------
    type
        Description of returned object.

    """
    #num_bands = len(np.unique(band_idx))
    Npt = 100
    tmin = -30
    tmax = 150
    num_bands = len(bands)
    GP_dict = {}

    # make our plots look nice
    stylePlots()

    for idx, row in df.iterrows():
        t = np.array(row["T"])
        f = np.array(row["Flux"])
        f[f<0.] = 0. #getting rid of negative flux

        #the magnitude-like array for the sake of the conversion
        y = np.log(f + 1)
        yerr = np.array(row["Flux_Err"]) / np.array(row["Flux"])
        t_test = np.linspace(tmin, tmax, Npt)
        band = row["Filter"]
        band_idx = pd.Series(row['Filter']).astype('category').cat.codes.values
        matrix = [t_test]

        def build_gp(params):
            time_kernel = tinygp.kernels.Matern32(jnp.exp(params["log_scale"]))
            kernel = Multiband(time_kernel, jnp.exp(params["log_diagonal"]), params["off_diagonal"])
            diag = yerr ** 2 + jnp.exp(2 * params["log_jitter"][X[1]])
            return tinygp.GaussianProcess(kernel, X, diag=diag, mean=lambda x: params["mean"][x[1]])

        #the GP parameters
        @jax.jit
        def loss(params):
            return -build_gp(params).condition(y)

        X = (t, band_idx)

        solver = jaxopt.ScipyMinimize(fun=loss)
        soln = solver.run(params)
        gp = build_gp(soln.params)

        df_t = []
        df_flux = []
        df_flux_err = []
        df_filt = []

        if idx%50 == 0:
            plt.figure(figsize=(10,7))

        for n in np.unique(band_idx):
            m = band_idx == n
            plt.errorbar(t[m], np.exp(y[m])-1,yerr=row['Flux_Err'][m], fmt="o", color=f"C{n}")
            mu, var = gp.predict(y, X_test=(t_test, np.full_like(t_test, n, dtype=int)), return_var=True)
            std = np.sqrt(var)

            if idx%50 == 0:
                plt.plot(t_test, np.exp(mu)-1, '.-', ms=2, color=f"C{n}")
                plt.fill_between(t_test,np.exp(mu - std)-1, np.exp(mu + std)+1, color=f"C{n}", alpha=0.3, label=bands[n])

            #going in order of band here--don't forget it!
            matrix.append(np.exp(mu)-1)
            matrix.append(std)

        if idx%50 == 0:
            plt.xlim((t_test[0], t_test[-1]))
            plt.xlabel("Phase from Trigger (Days)")
            plt.ylabel("Flux")
            plt.legend()
            plt.savefig(plotpath + "/GP_%i.png"%row.CID,dpi=200, bbox_inches='tight')

        stacked = np.vstack(matrix)
        GP_dict[row.CID] = stacked

    with open(savepath + '/%s_%i.pkl'%(fn, ts), 'wb') as f:
        pickle.dump(GP_dict, f)
    return GP_dict
