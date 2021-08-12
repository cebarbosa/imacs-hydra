""" Run paintbox in observed data. """
import os
import shutil
import copy

import numpy as np
from scipy import stats
import multiprocessing as mp
from astropy.table import Table, hstack, vstack
import astropy.constants as const
import emcee
import matplotlib.pyplot as plt
from tqdm import tqdm
from ppxf import ppxf_util
import seaborn as sns
import paintbox as pb
from paintbox.utils import CvD18, disp2vel

import context

def make_paintbox_model(wave, store, name="test", porder=45, nssps=1, sigma=100):
    """ Prepare a model with paintbox. """
    ssp = CvD18(sigma=sigma, store=store, libpath=context.cvd_dir)
    twave = ssp.wave
    limits = ssp.limits
    if nssps > 1:
        for i in range(nssps):
            p0 = pb.Polynomial(twave, 0, pname="w")
            p0.parnames = [f"w_{i+1}"]
            s = copy.deepcopy(ssp)
            s.parnames = ["{}_{}".format(_, i+1) for _ in s.parnames]
            if i == 0:
                pop = p0 * s
            else:
                pop += (p0 * s)
    else:
        pop = ssp
    vname = "vsyst_{}".format(name)
    stars = pb.Resample(wave, pb.LOSVDConv(pop, losvdpars=[vname, "sigma"]))
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder, zeroth=True, pname=f"p{name}")
    # Including emission lines
    target_fwhm = lambda w: sigma / const.c.to("km/s").value * w * 2.355
    gas_templates, gas_names, line_wave = ppxf_util.emission_lines(
        np.log(twave), [wave[0], wave[-1]], target_fwhm,
        tie_balmer=True, vacuum=True)
    gas_templates /= np.max(gas_templates, axis=0) # Normalize line amplitudes
    gas_names = [_.replace("_", "") for _ in gas_names]
    # for em in gas_templates.T:
    #     plt.plot(twave, em)
    # plt.show()
    if len(gas_names) > 0:
        emission = pb.NonParametricModel(twave, gas_templates.T,
                                         names=gas_names)
        emkin = pb.Resample(wave, pb.LOSVDConv(emission,
                            losvdpars=[vname, "sigma_gas"]))
        sed = (stars + emkin) * poly
    else:
        sed = stars * poly
    return sed, limits, gas_names

def set_priors(parnames, limits, linenames, vsyst, nssps=1):
    """ Defining prior distributions for the model. """
    priors = {}
    for parname in parnames:
        name = parname.split("_")[0]
        if name in limits:
            vmin, vmax = limits[name]
            delta = vmax - vmin
            priors[parname] = stats.uniform(loc=vmin, scale=delta)
        elif parname in vsyst:
            priors[parname] = stats.norm(loc=vsyst[parname], scale=500)
        elif parname == "eta":
            priors["eta"] = stats.uniform(loc=1., scale=19)
        elif parname == "nu":
            priors["nu"] = stats.uniform(loc=2, scale=20)
        elif parname == "sigma":
            priors["sigma"] = stats.uniform(loc=50, scale=300)
        elif parname == "sigma_gas":
            priors[parname] = stats.uniform(loc=50, scale=100)
        elif name == "w":
            priors[parname] = stats.uniform(loc=0, scale=1)
        elif name in linenames:
            priors[parname] = stats.expon(loc=0, scale=0.5)
        elif name in ["pred", "pblue"]:
            porder = int(parname.split("_")[1])
            if porder == 0:
                mu, sd = 1 / nssps, 1
                a, b = (0 - mu) / sd, (np.infty - mu) / sd
                priors[parname] = stats.truncnorm(a, b, mu, sd)
            else:
                priors[parname] = stats.norm(0, 0.05)
        else:
            print(f"parameter without prior: {parname}")
    return priors

def log_probability(theta):
    """ Calculates the probability of a model."""
    global priors
    global logp
    lp = np.sum([priors[p].logpdf(x) for p, x in zip(logp.parnames, theta)])
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    ll = logp(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_sampler(outdb, nsteps=5000):
    global logp
    global priors
    ndim = len(logp.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(logp.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    try:
        pool_size = context.mp_pool_size
    except:
        pool_size = 1
    pool = mp.Pool(pool_size)
    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                         backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    return

def weighted_traces(parnames, trace, nssps):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    weights = np.array([trace["w_{}".format(i+1)].data for i in range(
        nssps)])
    wtrace = []
    for param in parnames:
        data = np.array([trace["{}_{}".format(param, i+1)].data
                         for i in range(nssps)])
        t = np.average(data, weights=weights, axis=0)
        wtrace.append(Table([t], names=["{}_weighted".format(param)]))
    return hstack(wtrace)

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
        pmask = np.where(masks[i]==0, True, False)
        wave = waves[i][pmask]
        flux = fluxes[i][pmask]
        fluxerr = fluxerrs[i][pmask]
        models = np.zeros((n, len(wave)))
        y = np.percentile(models, 50, axis=(0,))
        for j in tqdm(range(len(trace)), desc="Generating models "
                                                         "for trace"):
            models[j] = seds[i](t[j])[pmask]
        y = np.percentile(models, 50, axis=(0,))
        yuerr = np.percentile(models, 84, axis=(0,)) - y
        ylerr = y - np.percentile(models, 16, axis=(0,))
        ax0 = fig.add_subplot(axs[0,i])
        ax0.errorbar(wave, flux, yerr=fluxerr, fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax0.plot(wave, y, c="tab:orange")
        ax0.xaxis.set_ticklabels([])
        ax0.set_ylabel("Flux")

        ax1 = fig.add_subplot(axs[1,i])
        ax1.errorbar(wave, 100 * (flux - y) / flux, yerr=100 * fluxerr, \
                                                                fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax1.plot(wave, 100 * (flux - y) / flux, c="tab:orange")
        ax1.set_ylabel("Res. (%)")
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

def plot_corner(trace, outroot, title=None, redo=False):
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Age": "Age (Gyr)", "x1": "$x_1$", "x2": "$x_2$", "Ca": "[Ca/H]",
              "Fe": "[Fe/H]", "Na": "[Na/H]",
              "K": "[K/H]", "C": "[C/H]", "N": "[N/H]",
              "Mg": "[Mg/H]", "Si": "[Si/H]", "Ca": "[Ca/H]", "Ti": "[Ti/H]"}
    title = "" if title is None else title
    output = "{}_corner.png".format(outroot)
    if os.path.exists(output) and not redo:
        return
    N = len(trace.colnames)
    params = trace.colnames
    data = np.stack([trace[p] for p in params]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    title = [title]
    for i, param in enumerate(params):
        parname = param.replace("_weighted", "")
        s = "{0}$={1:.2f}^{{+{2:.2f}}}_{{-{3:.2f}}}$".format(
            labels[parname], v[i], vuerr[i], vlerr[i])
        title.append(s)
    fig, axs = plt.subplots(N, N, figsize=(3.54, 3.5))
    grid = np.array(np.meshgrid(params, params)).reshape(2, -1).T
    for i, (p1, p2) in enumerate(grid):
        p1name = p1.replace("_weighted", "")
        p2name = p2.replace("_weighted", "")
        i1 = params.index(p1)
        i2 = params.index(p2)
        ax = axs[i // N, i % N]
        ax.tick_params(axis="both", which='major',
                       labelsize=4)
        if i // N < i % N:
            ax.set_visible(False)
            continue
        x = data[:,i1]
        if p1 == p2:
            sns.kdeplot(x, shade=True, ax=ax, color="C0")
        else:
            y = data[:, i2]
            sns.kdeplot(x, y, shade=True, ax=ax, cmap="Blues")
            ax.axhline(np.median(y), ls="-", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 16), ls="--", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 84), ls="--", c="k", lw=0.5)
        if i > N * (N - 1) - 1:
            ax.set_xlabel(labels[p1name], size=7)
        else:
            ax.xaxis.set_ticklabels([])
        if i in np.arange(0, N * N, N)[1:]:
            ax.set_ylabel(labels[p2name], size=7)
        else:
            ax.yaxis.set_ticklabels([])
        ax.axvline(np.median(x), ls="-", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 16), ls="--", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 84), ls="--", c="k", lw=0.5)
    plt.text(0.6, 0.7, "\n".join(title), transform=plt.gcf().transFigure,
             size=8)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.98, bottom=0.08,
                        hspace=0.04, wspace=0.04)
    fig.align_ylabels()
    for fmt in ["png", "pdf"]:
        output = "{}_corner.{}".format(outroot, fmt)
        plt.savefig(output, dpi=300)
    plt.close(fig)
    return

def run_paintbox(galaxy, spec, V0s, dlam=100, nsteps=5000, loglike="normal2",
                 nssps=1, target_res=None):
    """ Run paintbox. """
    global logp, priors
    target_res = [200, 100] if target_res is None else target_res
    # List of sky lines to be ignored in the fitting
    skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                         5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                         6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                         8747, 8757, 8767, 8777, 8787, 8797, 8827, 8836,
                         8919, 9310])
    dsky = 3 # Space around sky lines
    logps = []
    wranges = [[4000, 6680], [7800, 8900]]
    waves, fluxes, fluxerrs, masks, seds, linenames = [], [], [], [], [], []
    for i, side in enumerate(["blue", "red"]):
        # Locationg where pre-processed models will be stored for paintbox
        sigma = target_res[i]
        store = os.path.join(context.home_dir, "templates",
                             f"CvD18_sig{sigma}_{side}.fits")
        if not os.path.exists(store):
            # Compiling the CvD models
            velscale = sigma / 2
            wmin = wave.min() - 180
            wmax = wave.max() + 50
            twave = disp2vel([wmin, wmax], velscale)
            CvD18(twave, sigma=sigma, store=store, libpath=context.cvd_dir)
        # Reading the data
        tab = Table.read(spec, hdu=i+1)
        #  Normalizing the data to make priors simple
        norm = np.nanmedian(tab["flux"])
        wave = tab["wave"].data
        flux = tab["flux"].data / norm
        fluxerr = tab["fluxerr"].data / norm
        mask = tab["mask"]
        idx = np.where((wave < wranges[i][0]) | (wave > wranges[i][1]))[0]
        mask[idx] = 1
        # Masking all remaining locations where flux is NaN
        mask[np.isnan(flux * fluxerr)] = 1
        # Masking lines from Osterbrock atlas
        for line in skylines:
            idx = np.argwhere((wave >= line - dsky) &
                              (wave <= line + dsky)).ravel()
            mask[idx] = 1.
        # Defining polynomial order
        wmin = wave[mask==0].min()
        wmax = wave[mask==0].max()
        porder = int((wmax - wmin) / dlam)
        # Building paintbox model
        sed, limits, lines = make_paintbox_model(wave, store,
                              nssps=nssps, name=side, sigma=sigma,
                              porder=porder)
        logp = pb.Normal2LogLike(flux, sed, obserr=fluxerr, mask=mask)
        logps.append(logp)
        waves.append(wave)
        fluxes.append(flux)
        fluxerrs.append(fluxerr)
        masks.append(mask)
        seds.append(sed)
        linenames += lines
    # Make a joint likelihood for all sections
    logp = logps[0]
    for i in range(nssps - 1):
        logp += logps[i+1]
    # Making priors
    v0 = {"vsyst_blue": V0s[0], "vsyst_red": V0s[1]}
    priors = set_priors(logp.parnames, limits, linenames, vsyst=v0, nssps=nssps)
    # Perform fitting
    pb_dir = f"paintbox_nssps{nssps}_{loglike}_nsteps{nsteps}.h5"
    if not os.path.exists(pb_dir):
        os.mkdir(pb_dir)
    dbname = os.path.join(pb_dir, spec.replace(".fits", ".h5"))
    # Run in any directory outside Dropbox to avoid problems
    tmp_db = os.path.join(os.getcwd(), dbname)
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    outdb = os.path.join(wdir, dbname)
    if not os.path.exists(outdb):
        run_sampler(tmp_db, nsteps=nsteps)
        shutil.move(tmp_db, outdb)
    # Post processing of data
    if context.node in context.lai_machines: #not allowing post-processing @LAI
        return
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(nsteps * 0.9), flat=True, thin=100)
    trace = Table(tracedata, names=logp.parnames)
    if nssps > 1:
        ssp_pars = list(limits.keys())
        wtrace = weighted_traces(ssp_pars, trace, nssps)
        trace = hstack([trace, wtrace])
    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
    make_table(trace, outtab)
    # Plot fit
    outimg = outdb.replace(".h5", "_fit.png")
    plot_fitting(waves, fluxes, fluxerrs, masks, seds, trace, outimg,
                 skylines=skylines)
    # Make corner plot
    # Choose columns for plot
    cols_for_corner = [_ for _ in trace.colnames if _.endswith("weighted")]
    corner_table = trace[cols_for_corner]
    corner_file = outdb.replace(".h5", "_corner") # It will be saved in png/pdf
    plot_corner(corner_table, corner_file, title=galaxy, redo=False)

if __name__ == "__main__":
    galaxies = ["NGC4033", "NGC7144"]
    V0s = {"NGC7144": (1390, 1860), "NGC4033": (1617, 1617)}
    for galaxy in galaxies:
        wdir = os.path.join(context.home_dir, f"data/{galaxy}")
        os.chdir(wdir)
        specs = sorted([_ for _ in os.listdir(".") if _.endswith(
                        "bg_noconv.fits")])
        V0 = V0s[galaxy]
        for spec in specs:
            run_paintbox(galaxy, spec, V0, nssps=2)