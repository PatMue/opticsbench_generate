import logging
import copy

from numba import jit
import pyfftw
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage
from PIL import Image as PIL_IMG


if __package__ and "opticsbenchgen" in __package__:
    from opticsbenchgen.utils.utils import get_crop, soft_plot_close
else:
    from PSF.ZernPoly import ZernPoly,get_zemax_fringe_polynomials
    from opticsbenchgen.utils.utils import get_crop, soft_plot_close

logger = logging.getLogger(__name__)


# outside PSF class definitions -> module functions:
def binning_for_cupy(PSF_cpu:np.ndarray,pp_cpu:np.ndarray,scaling_factor:float=None,\
        specs:dict=None,verbose = False,binFactor:list=None,crop_to_size:list=None):
    """!
    Bin PSF (CPU implementation - comparison)
    """
    grid_mode = True if not PSF_cpu.shape[0] % 2 else False
    if binFactor is not False:
        PSF_cpu = ndimage.zoom(PSF_cpu, scaling_factor, output=None, \
            order=3, mode='constant', cval=0.0, prefilter=True,grid_mode=grid_mode)       
    if crop_to_size is None:
        targetsize = specs["camera_data"]["maximum_PSF_size_m"] # 50µm in [m]
        crop = get_crop(PSF_cpu.shape[0],targetsize_m=targetsize,\
            pixelsize=specs["camera_data"]["pixelSize"],odd=odd)
    else:
        crop = get_crop(PSF_cpu.shape[0],crop_to_size[0])
        logger.debug("crop_to_size: {}".format(crop_to_size))		
    if crop is not None:
        PSF_cpu = PSF_cpu[crop,crop]
    return PSF_cpu


#for speeding up the below code execution by 2-3 times:
#@jit(nopython=True) # TODO: not working here. 
def multiply_helper(pp:np.ndarray,W:np.ndarray,wave:float=None):
    """!
    Create complex wavefront function (2D)
    @param[in] pp: <np.ndarray> pupil transmission
    @param[in] W: <np.ndarray> pupil phase  in [waves] --> wave * 2*np.pi / wave = 2*np.pi
    @param[in] wave: <float> if W is in [m] instead of [wave], then scale with wave
    """
    if wave is None:
        return np.multiply(pp,np.exp(-1j*(+2*np.pi)*W)) # -1j according to Fringe convention
    return np.multiply(pp,np.exp(-1j*(+2*np.pi/wave)*W)) # -1j according to Fringe convention


#@jit(forceobj=True) # nopython not possible
def normalize_helper(psf:np.ndarray):
    """! Normalize the psf."""
    return psf/np.sum(np.sum(psf)) # normalize to 1 (energy norm, l1)


@jit(nopython=True)
def normalize_c_like_helper(psf:np.ndarray):
    """! Normalize, compile as C program using jit"""
    val = 0
    for i in range(0,psf.shape[0]):
        for j in range(0,psf.shape[1]):
            val += psf[i][j]
    return psf/val # normalize to 1 (energy norm, l1)


def get_fft2_helper(pyfftwObj,complex_pupil_array):
    """! propagate the pupil distribution"""
    return pyfftwObj(complex_pupil_array)


def get_normed_coordinate(yy,center):
    """! valid formula for both xx,yy"""
    return 2*(yy/(2*center-1)) -1


def get_zernike_ordering_nm_helper(j_max:int=15,ordering:str="Fringe",skip_zero:bool=True,**kwargs):
    """ 
    @param[in] j_max=15: maximum order for the chosen single indexing scheme
    @param[in] ordering="Fringe": indexing scheme to create j_max tuples: "Fringe","OSA/ANSI" or "Wyant"
    @param[out] nm: list of Zernike polynomials ordered with respect to selected indexing scheme
    @param[out] coeff: list of zeros matching the number of coefficients (for dummy values, Airy)
    """
    nm = []
    n = 0
    next_coeff = True
    if ordering in "OSA/ANSI":
        while next_coeff:
            m = -n
            while next_coeff and (m <= n):
                if len(nm) < j_max:
                    if not (skip_zero and ((n==0) and (m==0))):
                        nm.append((n,m))
                else: next_coeff = False
                m += 2
            n += 1
    elif ordering in "Fringe":
        nm,coeff = get_zernike_ordering_nm_helper(pow(j_max,2),ordering="OSA/ANSI")
        nm.sort(key=get_fringe_idx)
        if skip_zero:
            nm = nm[0:j_max-1]
        else:
            nm = nm[0:j_max]
    elif ordering in "Wyant":
        nm,coeff = get_zernike_ordering_nm_helper(j_max=j_max-1,ordering="Fringe")
    else:
        raise ValueError(f"Unknown ordering scheme {ordering}. Use: 'Fringe', 'OSA/ANSI', 'Wyant'.")

    coeff = list(np.zeros(len(nm)))

    return nm,coeff


def get_fringe_idx(elem):
    n = elem[0]
    m = elem[1]
    if m == 0:
        return pow(n/2 + 1,2)
    return pow((n+abs(m))/2 + 1,2) - 2*abs(m) + (1-np.sign(m))/2


