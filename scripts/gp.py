import jaxopt
import numpy as np
import pandas as pd
import tinygp
import jax
import jax.numpy as jnp
from io import StringIO
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)
bands = 'ugrizY'
N_bands = len(bands)

class Multiband(tinygp.kernels.Kernel):
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

def getGPLCs(df, savepath='./', num_bands=6):
    #num_bands = len(np.unique(band_idx))
    df['GP_T'] = df['T']
    df['GP_Filter'] = df['Filter']
    df['GP_Flux'] = df['Flux']
    df['GP_Flux_Err'] = df['Flux_Err']
    Npt = 100
    tmin = -30
    tmax = 150
    for idx, row in df.iterrows():
        print(idx)
        t = np.array(row["T"])
        f = np.array(row["Flux"])
        f[f<0] = 0 #getting rid of negative flux
        y = np.log(f + 1)
        yerr = np.array(row["Flux_Err"]) / np.array(row["Flux"])
        t_test = np.linspace(tmin, tmax, Npt)
        band = row["Filter"]
        band_idx = pd.Series(row['Filter']).astype('category').cat.codes.values

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

        if idx%100 == 0:
            plt.figure(figsize=(10,7))
        for n in np.unique(band_idx):
            m = band_idx == n
            plt.errorbar(t[m], np.exp(y[m])-1,yerr=row['Flux_Err'][m], fmt="o", color=f"C{n}")
            mu, var = gp.predict(y, X_test=(t_test, np.full_like(t_test, n, dtype=int)), return_var=True)
            std = np.sqrt(var)

            if idx%5 == 0:
                plt.plot(t_test, np.exp(mu)-1, 'o-', marker='.', ms=2, color=f"C{n}")
                plt.fill_between(t_test,np.exp(mu - std)-1, np.exp(mu + std)+1, color=f"C{n}", alpha=0.3)

            df_t.append(t_test)
            df_flux.append(np.exp(mu)-1)
            df_flux_err.append(std)
            df_filt.append([bands[n]]*len(mu))

        if idx%100 == 0:
            plt.xlim((t_test[0], t_test[-1]))
            plt.xlabel("Phase from Trigger (Days)")
            plt.ylabel("Flux")
            plt.savefig(savepath + "/GP_interpolation_%i.png"%row.CID,dpi=200, bbox_inches='tight')

        df.at[idx, 'GP_T'] =  np.concatenate(df_t)
        df.at[idx, 'GP_Flux'] = np.concatenate(df_flux)
        df.at[idx, 'GP_Flux_Err'] = np.concatenate(df_flux_err)
        df.at[idx, 'GP_Filter'] = np.concatenate(df_filt)
    return df
