# Patrick Müller, 2020-2023
"""!@package docstring
Class ZernPoly / zemax_fringe_polynomials
"""
from math import factorial as fact
import logging

import numpy as np
from numba import jit # just-in-time-compile functionality

logger = logging.getLogger(__name__)

class ZernPoly: # note: this is modified from Paul Fricker's zernfun implementation in Matlab
    """!
    ZernPoly: Computes Zernike Polynomials for different conventions or indexing schemes
    """
    def __init__(self,n,m,rho,phi):
        """!
        Initialize the ZernPoly class object
        """
        self.n = n         # number of zernike polynomial
        self.m = m
        self.rho = rho         # polar coordinates
        self.phi = phi

        raise NotImplementedError
    

def get_zemax_fringe_polynomials(rho,phi,jmax=15,skip_zero=True):
    """! verified on 15.07.2022 (again,updated) -- Zemax polynomials 'fringe' """

    # Wyant --> Fringe

    Z = [] #print(f"rho: {rho.shape}, phi: {phi.shape}") --> n_points
    jmin = 2 if skip_zero else 1 # skip piston Fringe 1
    logger.debug(f"using {jmin} to {jmax+1} polynomials, " + f"rho: {rho.shape},phi: {phi.shape}")
    for j in range(jmin,jmax + 1):
        if j == 1:
            Z.append(np.ones_like(rho))
        elif j == 2:
            Z.append(rho*np.cos(phi))
        elif j == 3:
            Z.append(rho*np.sin(phi))
        elif j == 4:
            Z.append(2*np.power(rho,2) - 1)
        elif j == 5:
            Z.append(np.power(rho,2) * np.cos(2*phi))
        elif j == 6:
            Z.append(np.power(rho,2) * np.sin(2*phi))
        elif j == 7:
            Z.append((3*np.power(rho,2) - 2) *  rho * np.cos(phi))
        elif j == 8:
            Z.append((3*np.power(rho,2) - 2) *  rho * np.sin(phi))
        elif j == 9:
            Z.append((6*np.power(rho,4) - 6*np.power(rho,2) + 1))
        elif j == 10:
            Z.append(np.power(rho,3) * np.cos(3*phi))
        elif j == 11:
            Z.append(np.power(rho,3) * np.sin(3*phi))
        elif j == 12:
            Z.append((4*np.power(rho,2) - 3) * np.power(rho,2) * np.cos(2*phi))
        elif j == 13:
            Z.append((4*np.power(rho,2) - 3) * np.power(rho,2) * np.sin(2*phi))
        elif j == 14:
            Z.append((10*np.power(rho,4) - 12*np.power(rho,2) + 3) * rho * np.cos(phi))
        elif j == 15:
            Z.append((10*np.power(rho,4) - 12*np.power(rho,2) + 3) * rho * np.sin(phi))
        elif j == 16:
            Z.append(20*np.power(rho,6) - 30*np.power(rho,4) + 12*np.power(rho,2) - 1)
        elif j == 17:
            Z.append(np.power(rho,4) * np.cos(4*phi))
        elif j == 18:
            Z.append(np.power(rho,4) * np.sin(4*phi))
        elif j == 19:
            Z.append((5*np.power(rho,2) - 4) * np.power(rho,3) * np.cos(3*phi))
        elif j == 20:
            Z.append((5*np.power(rho,2) - 4) * np.power(rho,3) * np.sin(3*phi))
        elif j == 21:
            Z.append((15*np.power(rho,4) - 20*np.power(rho,2) + 6) \
                *np.power(rho,2) * np.cos(2*phi))
        elif j == 22:
            Z.append((15*np.power(rho,4) - 20*np.power(rho,2) + 6) \
                *np.power(rho,2) * np.sin(2*phi))
        elif j == 23:
            Z.append((35*np.power(rho,6) - 60*np.power(rho,4) + 30*np.power(rho,2) -4)\
                * rho * np.cos(phi))
        elif j == 24:
            Z.append((35*np.power(rho,6) - 60*np.power(rho,4) + 30*np.power(rho,2) -4)\
                * rho * np.sin(phi))
        elif j == 25:
            Z.append(70*np.power(rho,8) - 140*np.power(rho,6) + 90*np.power(rho,4) \
                - 20*np.power(rho,2) + 1 )
        elif j == 26:
            Z.append(np.power(rho,5) * np.cos(5*phi))
        elif j == 27:
            Z.append(np.power(rho,5) * np.sin(5*phi))
        elif j == 28:
            Z.append((6*np.power(rho,2) - 5) * np.power(rho,4) * np.cos(4*phi))
        elif j == 29:
            Z.append((6*np.power(rho,2) - 5) * np.power(rho,4) * np.sin(4*phi))
        elif j == 30:
            Z.append((21*np.power(rho,4) - 30*np.power(rho,2) + 10) \
                *np.power(rho,3)*np.cos(3*phi))
        elif j == 31:
            Z.append((21*np.power(rho,4) - 30*np.power(rho,2) + 10) \
                *np.power(rho,3)*np.sin(3*phi))
        elif j == 32:
            Z.append((56*np.power(rho,6) - 105*np.power(rho,4) + 60*np.power(rho,2) - 10) \
                * np.power(rho,2) * np.cos(2*phi))
        elif j == 33:
            Z.append((56*np.power(rho,6) - 105*np.power(rho,4) + 60*np.power(rho,2) - 10) \
                * np.power(rho,2) * np.sin(2*phi))
        elif j == 34:
            Z.append((126*np.power(rho,8) - 280*np.power(rho,6) + 210*np.power(rho,4) \
                - 60*np.power(rho,2) + 5) * rho * np.cos(phi))
        elif j == 35:
            Z.append((126*np.power(rho,8) - 280*np.power(rho,6) + 210*np.power(rho,4) \
                - 60*np.power(rho,2) + 5) * rho * np.sin(phi))
        elif j == 36:
            Z.append((252*np.power(rho,10) - 630*np.power(rho,8) + 560*np.power(rho,6) \
                - 210*np.power(rho,4) + 30*np.power(rho,2) - 1))
        elif j == 37:
            Z.append((924*np.power(rho,12) - 2772*np.power(rho,10) + 3150*np.power(rho,8) \
                - 1680*np.power(rho,6) + 420*np.power(rho,4) - 42*np.power(rho,2) + 1))
        elif j == 38:
            Z.append(np.zeros_like(rho))  # dummy value 
    return Z


@jit(nopython=True)
def factorial_helper(n:int,f=1):
    for c in range(1,n+1):
        f *=c
    return f
#EOF
