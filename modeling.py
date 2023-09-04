# (c) Patrick Müller 2023

import os
import copy 
import argparse 
import functools
import json
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage 
from PIL import Image


from opticsbenchgen import psf_generator


__corruptions__ = ['defocus','vertical_astigmatism','oblique_astigmatism',
                   'horizontal_coma','vertical_coma','primary_spherical',
                   'oblique_trefoil','vertical_trefoil','defocus blur']

__severities__ = [1,2,3,4,5]
__colors__ = ['r','g','b']

__waves__ = [0.4861,0.5876,0.6563][::-1]  # 'b', 'g', 'r' --> rgb , in [µm]

# standard weights 
__weights__ = {
    1:1,
    2:4,
    3:4,
    4:3,
    5:6,
    6:6,
    7:8,\
    8:8,
    9:5,
    10:8,
    11:8,
    12:10,
    13:10,\
    14:12,
    15:12,
    16:7,
    17:10,
    18:10,
    19:12,\
    20:12,
    21:14,
    22:14,
    23:16,
    24:16,
    25:9,
    26:12,
    27:12,
    28:14,
    29:14,
    30:16,
    31:16,
    32:18,
    33:18,
    34:20,
    35:20,
    36:11,
    37:13
    }

__weights__ = {i: math.sqrt(v) for i,v in __weights__.items()}


def _fringe2standard(coeffs):
    return [(c,val*__weights__[c]) for c,val in coeffs]


def _standard2fringe(coeffs):
    return [(c,val/__weights__[c]) for c,val in coeffs]


def get_rms_from_standard(standard):
    vals2 = [v**2 for c,v in standard if c > 3] # rms to centroid 
    wavefront_rms = np.sqrt(np.sum(vals2)) # 'standard deviation' of ref and act sphere, for standard coeffs -> squared
    return wavefront_rms


class KernelAction():

    """
    Class to optimize kernel appearance    
    """
    
    def __init__(self,kernel,type='numpy'):
        """
        Base class to operate on a kernel 
        """
        self.kernel = kernel
        self.type = 'numpy'
        self.ndim = kernel.ndim


    @staticmethod    
    def rotate(kernel,k=1,istype='numpy'):
        """ rotates a kernel by 90° clockwise """
        if istype == 'numpy':
            if kernel.ndim == 3:
                rotated = np.zeros_like(kernel)
                for ch in range(3):
                    rotated[:,:,ch] = np.rot90(kernel[:,:,ch],k=k,axes=(0,1))
                return rotated
            return np.rot90(kernel,k=k,axes=(0,1))
            
        return torch.rot90(kernel,k=k,dims=[0,1])


    @staticmethod
    def set_center_of_mass_to_array_center(kernel,luminance=False):
        """
        takes a kernel and sets the center of mass to array center 
        """

        if not luminance: 
            if kernel.ndim == 3:
                for ch in range(3):
                    com = KernelAction._get_center_of_mass(kernel[:,:,ch])
                    ctr = KernelAction._get_array_center(kernel[:,:,ch])
                    shift = [round(m-c) for m,c in zip(com,ctr)] # integer shifts only
                    kernel[:,:,ch] = ndimage.shift(kernel[:,:,ch],[-s for s in shift],
                                                   mode='constant')         
                return kernel          
            
        
        com = KernelAction._get_center_of_mass(kernel,luminance=True)
        ctr = KernelAction._get_array_center(kernel,luminance=True)


        shift = [round(m-c) for m,c in zip(com,ctr)]
        if kernel.ndim == 3:
            color_dim = kernel.shape.index(3)
            if color_dim == 0:
               for ch in range(3):
                    kernel[ch,:,:] = ndimage.shift(kernel[ch,:,:],[-s for s in shift],
                                                   mode='constant')         
            else:
               for ch in range(3):
                    kernel[:,:,ch] = ndimage.shift(kernel[:,:,ch],[-s for s in shift],
                                                   mode='constant')         
            return kernel 
        

        return ndimage.shift(kernel,[-s for s in shift],mode='constant')    


    @staticmethod
    def _get_center_of_mass(kernel,luminance=True,num_digits=3):
        # data loading, does not require
        if kernel.ndim == 3 and not luminance:
            a,b,c = kernel.shape
            color_dim = [a,b,c].index(3)
            coms = []
            if color_dim == 2:
                for ch in range(3):
                    com = ndimage.center_of_mass(kernel[:,:,ch])
                    com = np.round(com,num_digits)
                    coms.append(com)
            if color_dim == 0:
                for ch in range(3):
                   com = ndimage.center_of_mass(kernel[ch,:,:])
                   com = np.round(com,num_digits)
                   coms.append(com)
            return coms 
        elif kernel.ndim == 3 and luminance:
            color_dim = kernel.shape.index(3)
            kernel = kernel.sum(axis=color_dim)
        return np.round(ndimage.center_of_mass(kernel),num_digits)


    @staticmethod
    def _get_array_center(kernel,luminance=True):
        ctr = list(kernel.shape)
        if kernel.ndim == 3:
            ctr.remove(3)
        ctr = [c //2 for c in ctr]  #quick implementation... 
        return ctr


    @staticmethod
    def show_two_images(im1,im2=None,ttl='',fontsize=12):
        
        if im2 is None:
            plt.imshow(np.clip(im1,0,1)/im1.max())
        else:
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(np.clip(im1,0,1)/im1.max())
            ax[1].imshow(np.clip(im2,0,1)/im2.max())
        plt.suptitle(ttl,fontsize=fontsize)
        plt.show()


class com_load(object):
    def __init__(self,func):
        self.func = func

    def __call__(self,*args,**kwargs):
        return KernelAction.set_center_of_mass_to_array_center(
            self.func(*args,**kwargs))
    
    def __get__(self,obj,obj_type):
        return functools.partial(self.__call__,obj)
    

class OpticsBenchGen(KernelAction):

    params = None 
    target_size = 25
    __l1_norm__ = True 


    def __init__(self,path_to_params):

        """
        l1 normalized rgb psfs used 
        """
        assert os.path.exists(path_to_params)

        self.paths = dict(path_to_params=path_to_params)
        self._load_params()
        self.__set_camera__()
        self.psfgen = psf_generator.get_psf  #alias 
        self.__all__ = [(par,sev) for par in self.params for sev in __severities__] # as iterator
        self.__len__ = len(self.__all__)


    def __set_camera__(self):
        self.camera_data = {
            "exit_pupil_Diameter":5e-3,
            "pp_factor":1, # not used
            "exit_pupil_sampling_grid_size":128,
            "sampling":128,
            "pixelSize":3.5e-6,\
            "focus_length":25e-3, #efl
            "wavelength_used":0.532e-6,\
            "sensor_px_x":100,"sensor_px_y":100,
            "maximum_PSF_size_m":100e-6
            }        
        

    def _load_params(self):
        """
        params in Wyant Scheme
        """
        with open(self.paths['path_to_params'],'r') as file:
            params = json.load(file)

        if "__chroma_factors__" in params:
            params.pop("__chroma_factors__")
        if "include_tilts" in params:
            params.pop("include_tilts")
        self.params = params 


    @com_load
    def __getitem__(self,n:int,sev:int=None,wfe:float=None):
        """
        severity: <int> from  [1,2,3,4,5]
        rgb: [0,1,2]
        #input(f"{n}, sev: {sev}: {wfe}")
        """ 

        if sev is not None:
            assert sev in __severities__
            wfe = self.params[str(n)]['values'][sev-1]
        else:
            assert wfe is not None 

        psf = np.zeros([self.target_size,self.target_size,3])
        for wave_id in range(3):
            wave = __waves__[wave_id]
            self.camera_data['wavelength_used'] = wave*1e-6 # [m]
            kernel = np.abs(self.psfgen(wyant_coeff=int(n),wfe=wfe,camera_data=self.camera_data,
                                      target_size=self.target_size)) # intensities
            if self.__l1_norm__:
                psf[:,:,wave_id] = kernel / kernel.sum() # l1 norm (default)
            else:
                psf[:,:,wave_id] = kernel
        return psf


def opticsbench_gen(args):
    """
    requires path to params (path_to_opticsbench_params)  *.json 
    to create OpticsBench Corruptions
    """

    if not args.path_to_opticsbench_params:
        args.path_to_opticsbench_params = 'params.json'

    opticsbench = OpticsBenchGen(args.path_to_opticsbench_params)

    kernels = np.zeros([8,5,25,25,3])
    for p, param in tqdm(enumerate(opticsbench.params),total=8,desc='Generate OpticsBench: '):
        if not param.__contains__('chroma_factors'):
            for s,sev in enumerate(__severities__):
                kernel = opticsbench.__getitem__(int(param),sev=sev)
                kernels[p,s,:,:,:] = kernel 

    if args.show:
        kernel1 = kernels[3,2,:,:]
        kernel2 = kernels[4,4,:,:]
        com1 = opticsbench._get_center_of_mass(kernel1)
        com2 = opticsbench._get_center_of_mass(kernel2)

        opticsbench.show_two_images(kernel1,kernel2,
                                    ttl=f'OpticsBench\n{com1},{com2}',fontsize=10)
    
    kernels = torch.moveaxis(torch.Tensor(kernels),-1,2)  # [8,5,3,25,25]
    return kernels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_opticsbench_params',type=str,default='',help='path where the psfs or params are')
    parser.add_argument('-s','--show',action='store_true',help='Show generated samples.')
    #parser.add_argument('--opticsbench',action='store_true',help='Generate the OpticsBench dataset')
    args = parser.parse_args()

    opticsbench_gen(args)