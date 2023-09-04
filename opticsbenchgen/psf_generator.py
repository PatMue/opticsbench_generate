# create basis psf. (zernike polynomials based) (c) Patrick Müller 2023
import os
import logging
import copy

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from skimage import transform
import numpy as np


if __package__ is not None and __package__.__contains__('opticsbenchgen'):
    from opticsbenchgen.PSF.PSF import PSF
    from opticsbenchgen.PSF.PSF import get_zernike_ordering_nm_helper 
else:
    from PSF.PSF import PSF


def rotate_psf_by_angle_in_degrees(psf:np.ndarray,angle:float=45,save=False):
    """!
    @param[in] psf <np.ndarray> , psf.ndim in [2,3]
    @param[in] angle <float>   0...360
    """
    psf =  transform.rotate(image=psf/psf.sum(axis=(0,1)),
                     angle= angle,#angle in degrees
                     resize=False,
                     center=None,
                     order=1,
                     mode="reflect",
                     cval=0,
                     clip=True,
                     preserve_range=False
                     )
    if save:
        savepath = input("Type the savepath: ")
        np.save(savepath,psf)
    return psf


def update_coeffs(coeffs:np.ndarray,coeff:int=3,val:float=1.0):
    """!"""
    coeffs[coeff] = val
    return coeffs


def get_psf(wyant_coeff:int=None,fringe_coeff:int=None,
            wfe:float=1.0,j_max:int=12,target_size=25,camera_data:dict=None): #-> np.ndarray:
    """    
    get a single psf with single coefficient     
    """
    if fringe_coeff:
        wyant_coeff = fringe_coeff -1  # internally uses wyant scheme
    
    psf_simple,coeffs = initialize_simple_psf(j_max=j_max,camera_data=camera_data,
                                              sz=target_size) 
    coeffs = update_coeffs(np.zeros_like(coeffs),coeff=wyant_coeff,val=wfe) 

    psf_simple = evaluate_psf(psf_simple,coeffs)
    psf_simple.binning(binFactor=True)
    psf_simple.crop()
    return psf_simple.PSF


def get_mixed_psf(wyant_coeffs:list=None,fringe_coeffs:list=None,
            wfes:list=None,j_max:int=12,target_size:int=25,camera_data:dict=None,
            skip_zero=True): #-> np.ndarray:
    """    
    get a single psf with multiple coefficients [(wyant_coeff_num,wfe),    
    skip_zero: <bool> True  -- opticsbenchgen default 
    """

    if fringe_coeffs:
        wyant_coeffs = [f-1 for f in fringe_coeffs]  # internally uses wyant scheme
    
    if skip_zero:
        wyant_coeffs = [w-1 for w in wyant_coeffs]

    psf_simple,coeffs = initialize_simple_psf(
                j_max=j_max,
                camera_data=camera_data,
                sz=target_size,
                skip_zero=skip_zero) 

    for wyant_coeff,wfe in zip(wyant_coeffs,wfes):
        coeffs = update_coeffs(coeffs,coeff=wyant_coeff,val=wfe) 

    psf_simple = evaluate_psf(psf_simple,coeffs)
    psf_simple.binning(binFactor=True)
    psf_simple.crop()
    return psf_simple.PSF


def get_basis_psfs(wfe=5.0,j_max=32):
    """!
    type in wyant scheme 
    """
    
    psf_simple,coeffs = initialize_simple_psf(j_max=j_max) #  nm.shape: (19,2)
    for coeff in range(5,12):#range(len(coeffs)):
        coeffs = update_coeffs(np.zeros_like(coeffs),coeff=coeff,val=wfe)
        psf_simple = evaluate_psf(psf_simple,coeffs)
        pupil = psf_simple.pp
        psf_simple.binning(binFactor=True)
        psf = psf_simple.PSF
        if coeff == 0:
            visualize_results(psf=psf,pupil=pupil,coeff=coeff)
        else:
            pupil = psf_simple.W * psf_simple.pp
            visualize_results(psf=psf,pupil=pupil,coeff=coeff)
    plt.close()


def evaluate_psf(psf_simple,coeffs):
    """"!"""
    psf_simple.getWavefront_using_precomputed_Z(coeffs)
    psf_simple.getPSF(showWarning=False)
    return psf_simple


def visualize_results(psf=None,pupil=None,coeff=None):
    """!"""
    if pupil is not None:
        plt.imshow(pupil)
        plt.title("Pupil simulation")
        soft_plot_close()
    if psf is not None:
        plt.imshow(psf)
        if coeff is not None:
            plt.title(f"coeff: {coeff}, Fringe: {coeff+1}, l1 [10^-3]: {np.round(np.einsum('ij ->',psf)*1e3,5)}")
        soft_plot_close()


def initialize_simple_psf(j_max=20,sz=25,camera_data=None,skip_zero=True):
    """!"""
    if camera_data is None:
        camera_data = {"exit_pupil_Diameter":5e-3,"pp_factor":1,\
            "exit_pupil_sampling_grid_size":128,"sampling":128,"pixelSize":3.5e-6,\
            "focus_length":25e-3,"wavelength_used":0.532e-6,\
            "sensor_px_x":100,"sensor_px_y":100,"maximum_PSF_size_m":100e-6}
        
    psf_simple = PSF(camera_data,interpolation_functions=None,
                    balance_coeff_fcns=None,accelerate_with_CUDA=False,\
                    j_max=j_max
                    )
    
    nm,__ = get_zernike_ordering_nm_helper(j_max=j_max,ordering="Fringe",skip_zero=skip_zero)
    coeffs = np.zeros(len(nm))
    logger.debug(f"nm ({np.shape(nm)}): {nm}")
    logger.debug(f"coeffs ({coeffs.shape})")
    psf_simple.crop_to_size = (sz,sz)
    return psf_simple, coeffs


def soft_plot_close(close=True):
    """!"""
    plt.show(block=False)
    input()
    if close:
        plt.close()


if __name__ == "__main__":
    logging_filename = "debug_psf_generator.log"
    logging_level = "DEBUG"
    logging.basicConfig(filename=logging_filename,level=logging_level) # using python 3.8.3
    get_basis_psfs()