""" Setup paintbox for running on example spectrum. """
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from astropy.table import Table
import astropy.constants as const
from spectres import spectres

import paintbox as pb

import context

def define_wranges(specs):
    """" Inspect masks, determine location of the chip gaps and determine
    wavelength ranges for the fit with paintbox. """
    masks = []
    for spec in specs:
        wave, flux, fluxerr, mask, res_kms = np.loadtxt(spec, unpack=True)
        mask = mask.astype(np.bool).astype(np.int)
        masks.append(mask)
    masks = np.array(masks)
    mask = masks.max(axis=0) # Combining all masks into one mask
    badpixels = np.array([i for i, x in enumerate(mask) if x == 0])
    sections0 = consecutive(badpixels)
    # Filling small gaps
    for section in sections0:
        gap_size = len(section)
        if gap_size < 50:
            mask[section] = 1
    goodpixels = np.array([i for i, x in enumerate(mask) if x == 1])
    sections = consecutive(goodpixels)
    w1 = [wave[x[0]] for x in sections]
    w2 = [wave[x[-1]] for x in sections]
    target_res = [250, 200, 200, 200, 100]
    wranges = Table([np.arange(len(w1))+1, w1, w2, target_res],
                  names=["section", "w1", "w2", "sigma_res"])
    wranges.write(os.path.join(context.home_dir, "tables/wranges.fits"),
                  overwrite=True)
    return wranges

def consecutive(data, stepsize=1):
    """ Groups regions of array with consecutive values.

    # https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def prepare_spectrum(spec_file, outspec, wranges, overwrite=False):
    """ Preparing the spectrum of a single galaxy for the fitting. """
    if os.path.exists(outspec) and not overwrite:
        return
    wave, flux, fluxerr, mask, res_kms = np.loadtxt(spec_file, unpack=True)
    mask = mask.astype(np.bool).astype(np.int)
    idx = np.where(mask > 0)[0]
    f_interp = interp1d(wave[idx], flux[idx], fill_value="extrapolate")
    flux = f_interp(wave)
    ferr_interp = interp1d(wave[idx], fluxerr[idx], fill_value="extrapolate")
    fluxerr = ferr_interp(wave)
    # Calculating resolution in FWHM
    c = const.c.to("km/s").value
    fwhms = res_kms / c * wave * 2.355
    # Homogeneize the resolution

    # Splitting the data to work with different resolutions
    names = ["wave", "flux", "fluxerr", "mask"]
    hdulist = [fits.PrimaryHDU()]
    for i, section in enumerate(wranges):
        w1 = section["w1"]
        w2 = section["w2"]
        target_res = section["sigma_res"]
        name = f"section{section['section']}"
        velscale = np.array(target_res / 3).astype(np.int)
        idx = np.where((wave >= w1) & (wave < w2))[0]
        w = wave[idx]
        f = flux[idx]
        ferr = fluxerr[idx]
        m = mask[idx]
        # res = res_kms[idx] # This was used to check a good target_res
        fwhm = fwhms[idx]
        target_fwhm = target_res / c * w * 2.355
        fbroad, fbroaderr = pb.broad2res(w, f, fwhm, target_fwhm, fluxerr=ferr)
        # Resampling data
        owave = pb.logspace_dispersion([w[0], w[-1]], velscale)
        oflux, ofluxerr = spectres(owave, w, fbroad, spec_errs=fbroaderr,
                                   fill=0, verbose=False)
        # Filtering the high variance of the output error for the error.
        ofluxerr = gaussian_filter1d(ofluxerr, 3)
        omask = spectres(owave, w, m, fill=0, verbose=False).astype(
            np.int).astype(np.bool)
        ########################################################################
        # Include mask for borders of spectrum
        wmin = owave[omask].min()
        wmax = owave[omask].max()
        omask[owave < wmin + 5] = False
        omask[owave > wmax - 5] = False
        # Mask for NaNs
        omask[np.isnan(oflux * ofluxerr)] = False
        ########################################################################
        obsmask = -1 * (omask.astype(np.int) - 1)
        table = Table([owave, oflux, ofluxerr, obsmask],
                      names=names)
        hdu = fits.BinTableHDU(table)
        hdu.header["EXTNAME"] = name.upper()
        hdu.header["SIGMA_RES"] = target_res
        hdulist.append(hdu)
    hdulist = fits.HDUList(hdulist)
    hdulist.writeto(outspec, overwrite=True)
    return

def prepare_sample_hydra():
    wdir = os.path.join(context.home_dir, "data")
    os.chdir(wdir)
    specs = sorted([_ for _ in os.listdir(wdir) if _.endswith("spectrum.dat")])
    pb_dir = os.path.join(context.home_dir, "paintbox")
    if not os.path.exists(pb_dir):
        os.mkdir(pb_dir)
    wranges = define_wranges(specs)
    for spec in specs:
        name = "_".join(spec.split("_")[:-1])
        spec_dir = os.path.join(pb_dir, name)
        if not os.path.exists(spec_dir):
            os.mkdir(spec_dir)
        outspec = os.path.join(spec_dir, f"{name}.fits")
        prepare_spectrum(spec, outspec, wranges, overwrite=True)

if __name__ == "__main__":
    prepare_sample_hydra()