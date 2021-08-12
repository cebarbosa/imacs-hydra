""" Perform analysis of results from paintbox. """
import os

import numpy as np
from astropy.table import Table, vstack, hstack
import matplotlib.pyplot as plt
from tqdm import tqdm
import emcee
import arviz as az
import paintbox as pb

import context

def weighted_traces(trace, sed, weights, outtab, redo=False):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    if os.path.exists(outtab) and not redo:
        a = Table.read(outtab)
        return a
    ws, ps = [], []
    for i in range(sed.nssps):
        w = trace[:, sed.parnames.index("w_{}".format(i+1))]
        j = [sed.parnames.index("{}_{}".format(p, i+1)) for p in
             sed.sspcolnames]
        ssp = trace[:, j]
        n = np.array([weights(s)[0] for s in ssp])
        ws.append(w * n)
        ps.append(ssp)
    ws = np.stack(ws)
    ws /= ws.sum(axis=0)
    ps = np.stack(ps)
    a = np.zeros((len(sed.sspcolnames), len(ws[1])))
    for i, param in enumerate(sed.sspcolnames):
        a[i] = np.sum(ps[:,:,i] * ws, axis=0)
    a = Table(a.T, names=sed.sspcolnames)
    a.write(outtab, overwrite=True)
    return a

def make_table(trace, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def plot_fitting(waves, fluxes, fluxerrs, masks, seds, trace, output,
                 skylines=None, dsky=3):
    width_ratios = [w[-1]-w[0] for w in waves]
    fig, axs = plt.subplots(2, len(seds), gridspec_kw={'height_ratios': [2, 1],
                            "width_ratios": width_ratios},
                            figsize=(2 * context.fig_width, 3))
    for i in range(len(waves)):
        sed = seds[i]
        t = np.array([trace[p].data for p in sed.parnames]).T
        n = len(t)
        wave = waves[i][masks[i]]
        flux = fluxes[i][masks[i]]
        fluxerr = fluxerrs[i][masks[i]]
        models = np.zeros((n, len(wave)))
        y = np.percentile(models, 50, axis=(0,))
        for j in tqdm(range(len(trace)), desc="Generating models "
                                                         "for trace"):
            models[j] = seds[i](t[j])[masks[i]]
        y = np.percentile(models, 50, axis=(0,))
        yuerr = np.percentile(models, 84, axis=(0,)) - y
        ylerr = y - np.percentile(models, 16, axis=(0,))
        ax0 = plt.subplot(axs[0,i])
        ax0.errorbar(wave, flux, yerr=fluxerr, fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax0.plot(wave, y, c="tab:orange")
        ax0.xaxis.set_ticklabels([])
        ax0.set_ylabel("Flux")

        ax1 = plt.subplot(axs[1,i])
        ax1.errorbar(wave, 100 * (flux - y) / flux, yerr=100 * fluxerr, \
                                                                fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax1.plot(wave, 100 * (flux - y) / flux, c="tab:orange")
        ax1.set_ylabel("Res. (\%)")
        ax1.set_xlabel("$\lambda$ (Angstrom)")
        ax1.set_ylim(-5, 5)
        ax1.axhline(y=0, ls="--", c="k")
        # Include sky lines shades
        if skylines is not None:
            for ax in [ax0, ax1]:
                w0, w1 = ax0.get_xlim()
                for skyline in skylines:
                    if (skyline < w0) or (skyline > w1):
                        continue
                    ax.axvspan(skyline - 3, skyline + 3, color="0.9",
                               zorder=-100)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)
    return

def weighted_traces(trace):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    nssps = len([_ for _ in trace.colnames if _.startswith("w_")])
    parnames = ["_".join(_.split("_")[:-1]) for _ in trace.colnames if
                         _.endswith("_1") and _ != "w_1"]
    weights = np.array([trace["w_{}".format(i+1)].data for i in range(nssps)])
    wtrace = []
    for param in parnames:
        data = np.array([trace["{}_{}".format(param, i+1)].data
                         for i in range(nssps)])
        t = np.average(data, weights=weights, axis=0)
        wtrace.append(Table([t], names=["{}_weighted".format(param)]))
    return hstack(wtrace)

def postprocessing(galaxy, nssps=1, loglike="studt2", nsteps=5000):
    """ Process fitting results from paintbox fitting. """
    wdir = os.path.join(context.home_dir, f"data/testdata/paintbox")
    dbname = f"{galaxy}_nssps{nssps}_{loglike}_nsteps{nsteps}.h5"
    outdb = os.path.join(wdir, dbname)
    # Load database and make a table with summary statistics



if __name__ == "__main__":
    galaxy = "NGC7144"
    postprocessing(galaxy)