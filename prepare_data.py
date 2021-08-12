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

from paintbox.utils import broad2res, disp2vel

import context

def prepare_spectrum(spec_file, outfile, overwrite=False):
    """ Preparing the spectrum of a single galaxy for the fitting. """
    if os.path.exists(outfile) and not overwrite:
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
    target_res = np.array([200, 100]) # Rounding up the ideal resolution
    velscale = (target_res / 3).astype(np.int)
    # Splitting the data to work with different resolutions
    wave_ranges = [[4200, 6680], [8200, 8900]]
    names = ["wave", "flux", "fluxerr", "mask"]
    hdulist = [fits.PrimaryHDU()]
    for i, (w1, w2) in enumerate(wave_ranges):
        idx = np.where((wave >= w1) & (wave < w2))[0]
        w = wave[idx]
        f = flux[idx]
        ferr = fluxerr[idx]
        m = mask[idx]
        # res = res_kms[idx] # This was used to check a good target_res
        fwhm = fwhms[idx]
        target_fwhm = target_res[i] / c * w * 2.355
        fbroad, fbroaderr = broad2res(w, f, fwhm, target_fwhm, fluxerr=ferr)
        # Resampling data
        owave = disp2vel([w[0], w[-1]], velscale[i])
        oflux, ofluxerr = spectres(owave, w, fbroad, spec_errs=fbroaderr)
        # Filtering the high variance of the output error for the error.
        ofluxerr = gaussian_filter1d(ofluxerr, 3)
        omask = spectres(owave, w, m).astype(np.int).astype(np.bool)
        ########################################################################
        # Include mask for borders of spectrum
        wmin = owave[omask].min()
        wmax = owave[omask].max()
        omask[owave < wmin + 5] = False
        omask[owave > wmax - 5] = False
        ########################################################################
        obsmask = -1 * (omask.astype(np.int) - 1)
        table = Table([owave, oflux, ofluxerr, obsmask],
                      names=names)
        hdu = fits.BinTableHDU(table)
        hdulist.append(hdu)
    hdulist = fits.HDUList(hdulist)
    hdulist.writeto(outfile, overwrite=True)
    return

def prepare_sample(sample, overwrite=False):
    for galaxy in sample:
        wdir = os.path.join(context.home_dir, "data", galaxy)
        os.chdir(wdir)
        spec_files = [_ for _ in os.listdir(wdir) if _.endswith("noconv.txt")]
        for spec_file in spec_files:
            outfile = os.path.join(wdir, spec_file.replace(".txt", ".fits"))
            prepare_spectrum(spec_file, outfile, overwrite=overwrite)

if __name__ == "__main__":
    galaxies = ["NGC4033", "NGC7144"]
    prepare_sample(galaxies, overwrite=True)