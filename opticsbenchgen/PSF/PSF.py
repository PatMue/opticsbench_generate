"""@package docstring
Module PSF -- generate psf from zernike polynomials. 
(c) Patrick Müller 2022-2026
"""
import logging
import copy
import numpy as np
from scipy import ndimage

if __package__ and "opticsbenchgen" in __package__:
    from opticsbenchgen.PSF.ZernPoly import get_zemax_fringe_polynomials
    from opticsbenchgen.utils.utils import get_crop, soft_plot_close
    from opticsbenchgen.PSF.utils import get_zernike_ordering_nm_helper
    import opticsbenchgen.PSF.geometry
else:
    from PSF.ZernPoly import get_zemax_fringe_polynomials
    from utils.utils import get_crop, soft_plot_close
    import PSF.geometry


logger = logging.getLogger(__name__)


class PupilGen():
    """!
    Generates a PSF from Zernike Polynomials:
    * all units should be physical  [m] , [1/m]  or pixel
    * wavelength: [m]
    * no optimization since only few verification psfs generated
    (if verified later --> compare with optimized version from framework)

    / based on prysm/poppy.  
    https://github.com/spacetelescope/poppy/blob/develop/poppy/geometry.py  s
    """
    dx = None
    dxp = None
    pupilgridsize = None
    imsize = None
    camera = None


    def __init__(self,camera:dict=None,sampling:int=2**8,zero_padding:bool=False,\
            as_api=True,verbose=False,**physical_dim):
        """!
            dx: <float> , spacing in psf space [m]
            dxp: <float>, spacing in pupil space [m]
            pupilgridsize: <float>, size of pupil in [m]
            imsize: <float>, size of diffraction image [m]
            camera: dict from specs
        """
        assert not sampling % 2, "Needs to be even sized"
        self.sampling = sampling
        self.zero_padding = zero_padding
        self.verbose = verbose
        self.log = logger.info if not self.verbose else print
        camera = self._wrap_camera_names(camera) if as_api else camera
        self._update_camera(camera)
        self.get_physical_grid_from(physical_dim)
        self.get_pupil_function()
        self._summarize()


    def _wrap_camera_names(self,camera:dict):
        """!"""
        try:
            cam = copy.deepcopy(camera)
            return {\
                "sampling":cam["exit_pupil_sampling_grid_size"],\
                "xp_diameter":cam["exit_pupil_Diameter"],\
                "xp_distance":cam["focus_length"],\
                "wave":cam["wavelength_used"]}
        except Exception as err:
            print(err)
            return None


    def _update_camera(self,camera:dict=None):
        """!
        overwrites only values which are within
        """
        fcn = logger.debug if not self.verbose else print
        if self.camera is None:
            cam = self._get_sample_camera()
            camera["sampling"] = self.sampling
        else:
            cam = self.camera
        camera = camera if camera is not None else {}
        for key in camera:
            if key in cam:
                fcn("replacing key {}: {} -> {}".format(key,cam[key],camera[key]))
                cam[key] = camera[key]
        self.camera = cam


    def _get_sample_camera(self):
        """!"""
        return {"dx":1e-6,"sampling":64,"wave":0.4861e-6,"imsize":0.1e-3,\
            "xp_diameter":6.392825e-3,"xp_distance":25.00002e-3}


    def _get_options(self):
        """!"""
        return ["dx","dxp","pupilgridsize","imsize"]


    def __verifyattr__(self,physical_dim):
        """!"""
        attr = [a for a in self._get_options() if physical_dim.__contains__(a)]
        self.physical = bool(len(attr))
        self.attr = attr[0] if self.physical else None


    def get_physical_grid_from(self,physical_dim:dict):
        """!
        If there is any of the supported kwargs given, model uses physical units
        args:
            dx: <float> , spacing in psf space [m]
            dxp: <float>, spacing in pupil space [m]
            pupilgridsize: <float>, size of pupil in [m]
            imsize: <float>, size of diffraction image [m]
        """
        assert len(physical_dim) == 1, "Exact one quantity required for setup"
        self.__verifyattr__(physical_dim)
        if self.physical:
            attr = self.attr
            report_fcn = logger.info if not self.verbose else print
            report_fcn(f"using {attr} to generate grid...")
            self.__setattr__(attr,physical_dim[attr])

            N = self.sampling
            d_xp = self.camera["xp_diameter"] # [m]
            z_xp = self.camera["xp_distance"] # [m]
            wave = self.camera["wave"]
            wz = wave * z_xp

            if attr == "dx":
                self.dxp = wz / (N * self.dx)
                self.imsize = N * self.dx
                self.pupilgridsize = wz / self.dx
            elif attr == "dxp":
                self.dx = wz / (N * self.dxp)
                self.imsize = wz / self.dxp
                self.pupilgridsize = N * self.dxp
            elif attr == "pupilgridsize":
                self.dx = wz / self.pupilgridsize
                self.dxp = self.pupilgridsize / N
                self.imsize = N * wz / self.pupilgridsize
            elif attr == "imsize":
                self.dx = self.imsize / N
                self.dxp = wz / self.imsize
                self.pupilgridsize = N * wz / self.imsize
            else:
                raise ValueError(f"attribute does not exist: {attr}")

            self.du = 1 / self.pupilgridsize
            self.fnum = z_xp / d_xp
            self.cutoff = 1 / (wave * self.fnum)
            self.k = 2*np.pi / wave
            self.wave_z0 = wz
            self.wave_z0_r_xp = 2* wz / d_xp
            self.first_airy_zero_calculated = 1.22*wave*z_xp / d_xp

            sz = 1e6 * self.dx * self.sampling / 2
            sz_p = 1e3 * self.dxp * self.sampling / 2

            self.extent = {"imsize":{"val":(-sz,sz,-sz,sz),"unit":"µm"},\
                "pupilgridsize":{"val":(-sz_p,sz_p,-sz_p,sz_p),"unit":"mm"}}

            self._update_camera({"dx":self.dx,"imsize":self.imsize})

        else:
            self.extent = (-1,1,-1,1)
            logger.info("\nnon physical computation... using normalized coords*******\n")

        self._save_properties()


    def _save_properties(self):
        """!"""
        self.properties = copy.deepcopy(self.__dict__)


    def get_pupil_function(self,use_module="framework"):
        """! Calculates the pupil function.
            @param[in] sampling: $2**9$
            @param[out] self.cart2pol(X,Y): the polar coordinates of the generated pupil

            Creates a square grid of size of the exit pupil diameter and uses
            this information as well as z0, wave and N for physical calculation
        """
        if not self.physical:
            self.x = np.arange(-1,1,self.sampling*2) #<-> self.x = np.linspace(-1,1,sampling)
            X,Y = np.meshgrid(self.x,self.x) # square aperture grid
            self.rho,self.phi = self.cart2pol(X,Y)
            self.pp_mask = self.rho <= 1
            self.pp = np.zeros(X.shape) # x^2 + y^2 = 1  --> equation for unit circle (plane)
            self.pp[np.hypot(self.x[np.newaxis,:],self.x[:,np.newaxis]) <= 1] = 1
            self.pupil_diameter_px = self.pp_mask[self.pp_mask[self.pp_mask.shape[0]//2]==True,:].shape[0]
        else:
            r_xp = self.camera["xp_diameter"] /2 # [m]
            xp = np.arange(-(self.sampling/2)*self.dxp,+(self.sampling/2)*self.dxp,self.dxp)
            xi,eta = np.meshgrid(xp,xp)
            self.rho,self.phi = self.cart2pol(xi,eta)
            self.pp = np.zeros_like(xi)
            radius = r_xp
            self.pp = (self.rho <= (r_xp - self.dxp)).astype(float)
            self.pp_mask = self.pp >= 0.99 #0.49 # 0.49 #0.749
            self.pupil_diameter_px = \
                self.pp_mask[self.pp_mask[self.pp_mask.shape[0]//2] == True,:].shape[0]
            self.xp = xp

        self.W = np.zeros_like(self.pp)


    def _summarize(self):
        """!"""
        if self.physical:
            fcn = self.log
            fcn("\n...........\nThe following setup is used:")
            fcn("D: {}mm\nz0: {}mm\nwave: {}µm\nN: {}\n".format(\
                self.camera["xp_diameter"]*1e3,self.camera["xp_distance"]*1e3,\
                self.camera["wave"]*1e6,self.sampling))
            fcn("Sampling:\ndxp: {}µm,\ndx: {}µm,\npupilgridsize: {}mm,\nimsize: {}mm\n\n".format(\
                self.dxp*1e6,self.dx*1e6,self.pupilgridsize*1e3,self.imsize*1e3))
            fcn("Camera used:")
            for item in self.camera.items():
                fcn("{}".format(item))
            fcn("...........\n")


    def _get_n_m_from_nm(self,nm=None,jmax=15,ordering="OSA/ANSI"):
        """! polynomial names """
        if nm is None:
            nm,__ = get_zernike_ordering_nm_helper(j_max=jmax,ordering=ordering)
        n = np.array([elem[0] for elem in nm])
        m = np.array([elem[1] for elem in nm])
        return n,m


    def __get_precomputed_zernike_polynomials_from_zemax__(self,jmax):
        """! evaluate on the (unit circle), therefore divide by the max_radius"""
        self.Z = get_zemax_fringe_polynomials(self.rho.flatten()/(\
            0.5*self.camera["xp_diameter"]),self.phi.flatten(),jmax=jmax)
        logger.debug("generated polynomials using explicit zemax expression: {}".format(\
            np.shape(self.Z)))


    def _get_aggregated_wavefront(self,coeff:np.ndarray,n:np.ndarray=None,m:np.ndarray=None):
        """!
        Compute the wavefront using Zernike polynomials:
            @param[in] coeff: [np.ndarray]  the coefficient values
            @param[in] n: [np.ndarray] the polynomial index
            @param[in] m: [np.ndarray] the polynomial index
        """
        self.W = np.reshape(np.squeeze(np.dot(coeff.transpose(),self.Z))[::-1],self.W.shape)#.transpose())[::-1]
        self.W = self.pp * self.W  #+ np.mean(self.W[self.pp_mask])
        logger.debug("np.mean W: {}".format(np.mean(self.W[self.pp_mask]))) # 1.56..


    def _create_imager(self):
        """!
        Create the specified imager.
        """
        pixelSize = self.camera["pixelSize"]
        imsize = self.camera["imsize"]
        sensor_dim = [round(pixelSize*sz,2) for sz in imsize]
        self.imager_center = [imsize[0]//2,imsize[1]//2]

        self.log("----------------\nphysical grid for imager created")
        self.log("camera_sensor_dim: [height:y,width:x] " + str(sensor_dim*1e3) + " mm")


    def _get_fringe_aggregated_wavefront_without_tilt(self,coeff):
        """!"""
        tilt_xy_removed = copy.deepcopy(coeff[1:3])
        coeff_temp = copy.deepcopy(coeff)
        coeff_temp[1:3] = 0 # set to zero in storage for the copied list
        self.log(f"Removing tilt x and y: {tilt_xy_removed} . Total of {len(coeff)} coefficients.")
        self._get_aggregated_wavefront(coeff_temp)
        return tilt_xy_removed


    def cart2pol(self,x:float,y:float,dt=None): # only used for pupil simulation (no rotation of PSFs)
        """!
        Center the grid if it is even --> dt/2
        Compute the polar coordinates from cartesian coordinates:
            @param[in] x: [float]
            @param[in] y: [float]
            @param[out] rho: [float] radius
            @param[out] phi_rad: [float] angle
        """
        if dt is None:
            dt = self.dxp
        rho = np.sqrt((x + dt/2)**2 + (y + dt/2)**2)  # magnitude
        phi_rad = np.arctan2(+y,x) # it's okay!
        return rho,phi_rad


    def pol2cart(self,rho:float,phi_rad:float,dt=None):
        """!
        Compute the cartesian coordinates from polar coordinates:
            @param[in] rho: [float] radius
            @param[in] phi_rad: [float] angle
            @param[out] x: [float]
            @param[out] y: [float]
        """
        if dt is None:
            dt = self.dxp
        x = rho * np.cos(phi_rad) - dt/2
        y = rho * np.sin(phi_rad) - dt/2
        return x,y


class PSF():
    """
    Apply Zernike polynomials to form a PSF
    """

    def __init__(self,camera_data:dict=None,interpolation_functions:list=None,\
            balance_coeff_fcns:list=None,scale_by_pixel_size:bool=True,\
            accelerate_with_CUDA:bool=False,j_max=20,verbose=False):
        """! Initializes the PSF class.
        @param camera_data: [dict] with all camera definitions
        @param interpolation_functions: [list] with coefficient fcns for interpolated imager grid
        @param balance_coeff_fcns: [list] containing parameter variations
        @param scale_by_pixel_size: [bool]
        @param accelerate_with_CUDA: [bool] whether to use GPU or CPU processing
        """

        """
        Initialize the PSF simulation based on pupil simulation
        """
        self.z = np.array([]) # zernike polynomials
        self.W = np.array([])  # phase error
        self.psf = np.array([])
        self.rho = np.array([]) # polar coordinates
        self.phi = np.array([])
        self.idx = np.array([])
        self.pp = np.array([]) # pupil function

        # additional attributes (pupil and imager definition):
        self.x = None
        self.delta_x = None
        self.df = None
        self.fx = None
        self.pp_pupil_size_px = None
        self.pp_pixel_size = None
        self.pp_radius = None
        self.relative_pupil_area = None
        self.physical_pixel_size = None
        self.physical_PSF_extent = None
        self.scaling_factor = None
        self.imager_resolution = None
        self.imager_center = None
        self.crop_to_size = None

        if camera_data is None:
            camera_data = {"exit_pupil_sampling_grid_size": 2**10,"pp_factor": 1}
        if isinstance(camera_data["pp_factor"],str):
            camera_data["pp_factor"] = float(camera_data["pp_factor"]) # should be 1.0
        self.pp_factor = camera_data["pp_factor"]
        self.camera_data = camera_data
        self.interpolation_functions = interpolation_functions
        self.balance_coeff_fcns= balance_coeff_fcns
        self.scale_by_pixel_size = scale_by_pixel_size
        self.j_max = j_max
        self.verbose = verbose
        self.log = logger.info if not self.verbose else print
        self.generate_pupil()
        logger.debug("initialized psf object with pupil {}".format(self.pp.shape))


    def generate_pupil(self,circular_pupil_sampling_min=32):
        """!
        generates a physical pupil.
        """
        camera_data = self.camera_data
        valid_sampling = False
        pupilgridsize = camera_data["exit_pupil_Diameter"]
        circular_pupil_sampling = copy.deepcopy(camera_data["exit_pupil_sampling_grid_size"])
        while not valid_sampling and circular_pupil_sampling >= circular_pupil_sampling_min:
            pupil = PupilGen(camera=camera_data,\
                sampling=camera_data["exit_pupil_sampling_grid_size"],\
                verbose=self.verbose,**{"pupilgridsize":pupilgridsize})
            self.scaling_factor = pupil.dx / camera_data["pixelSize"]
            logger.debug("scaling factor: {}".format(self.scaling_factor))
            if self.scaling_factor >= 1 and not valid_sampling:
                msg = "Physical psf sampling {}µm is bigger than ".format(\
                    np.round(pupil.dx*1e6,3))
                msg += "the imager pixel size {}µm. Decrease wave*z_xp / pupilgridsize".format(\
                np.round(camera_data["pixelSize"]*1e6,3))
                pupilgridsize *= 2
                circular_pupil_sampling /= 2
                msg += f"\nDoubling pupilgridsize to: {np.round(1e3*pupilgridsize,3)}mm."
                msg += f"Circular pupil: {circular_pupil_sampling}/{pupil.sampling} values."
                logger.warning(msg)
                self.log(msg)
            else:
                valid_sampling = True
                pupil.__get_precomputed_zernike_polynomials_from_zemax__(self.j_max)
                self.Z = pupil.Z
                self.__properties__ = pupil.properties
                self.physical_extent = pupil.extent
                self.physical_pixel_size = pupil.dx  # psf sampling [m]
                self.pp_mask = pupil.pp_mask
                self.pp = pupil.pp
                self.df  = pupil.dxp # pupil sampling
                self.simulated_pp_diameter_px = \
                    self.pp_mask[self.pp_mask[self.pp_mask.shape[0]//2] == True,:].shape[0]
                
                self.log(f"Scaling {pupil.dx} to {camera_data['pixelSize']}, {self.scaling_factor}.")

        assert circular_pupil_sampling >= 32, "Increase sampling to allow for processing."
        self.pupilgen = pupil


    def getWavefront_using_precomputed_Z(self,coeff:np.ndarray):
        """!
        Use precomputed single Zernike Polynomials to acquire the wavefront
        @param[in] coeff: [np.ndarray]  the coefficient values
        """
        logger.debug("this function can only be used if all wavefronts have the same" +\
            "(number) of polynomials required <-> n_list_ij == n_list for m respectively)")
        logger.debug("shape of coeff: {}, Z: {}, pp,W: {}".format(coeff.shape,np.shape(self.Z),\
            self.pp.shape))
        self.W = np.reshape(np.squeeze(np.dot(coeff.transpose(),self.Z).transpose()),self.pp.shape)


    def getPSF(self,showWarning=True):
        """!
        Deprecated. Use pyfftw.
        """
        if showWarning:
            print('deprecated: getPSF_using_pyfftw() is much faster')
        P = np.multiply(self.pp,np.exp(-1j*(+2)*np.pi*self.W))
        self.psf = np.abs(np.fft.fftshift(np.fft.fft2(P),axes=None)*self.df**2)**2
        self.psf = self.psf/self.psf.sum()


    def getPSF_using_pyfftw(self,pyfftwObj):
        """
        Get the psf using pyfftw acceleration.
        @[in] pyfftwObj: Initialized in Dataflow class once for use in for loop to reduce overhead
        """
        # https://github.com/brandondube/prysm/blob/master/prysm/propagation.py
        logger.debug("W: {}, pupil: {}, df: {}".format(self.W.shape,self.pp.shape,self.df))
        self.psf = np.abs(np.fft.fftshift(pyfftwObj(np.multiply(\
            self.pp,np.exp(-1j*(+2*np.pi)*self.W))),axes=None)*self.df**2)**2
        self.psf = self.psf / self.psf.sum() # 0...1

 
    def rotate_using_ndimage(self,x,y):
        """!
        Rotate PSF using scipy.ndimage.rotate (for consistency checks)
        """
        angle = np.rad2deg(-np.arctan2((y-self.imager_center[0]),(x-self.imager_center[1]))) -90
        # fixed to +angle --> using validation proof
        if angle >= 180:
            angle = 180 - angle
        self.psf = ndimage.rotate(self.psf,
                            angle,
                            axes=(1, 0),
                            reshape=False,
                            output=None,
                            order=3,
                            mode='constant',
                            cval=0.0,
                            prefilter=True)
        return self.psf/np.max(self.psf) # 0...1


    def binning(self, binFactor:list=None,order=3,mode='constant'):
        """
        @param[in] binFactor:   Target size to resample the PSF to.
                                None: no binning,
                                Empty list []: automatic (use physical representation) -> default
                                [0.5]: user scaling factor
                                
                                (order = 1 as experiments suggested on 17.02.2023)
        """
        if binFactor is not False:
            grid_mode = True if not self.psf.shape[0] % 2 else False
            self.psf = ndimage.zoom(self.psf, self.scaling_factor, output=None, \
                                        order=order, mode=mode, cval=0.0, 
                                        prefilter=True,grid_mode=grid_mode
                                        )
            
                                         
    def crop(self,odd=True):
        """ crop the psf to physical size / target size."""
        if self.crop_to_size is None:
            targetsize = self.camera_data["maximum_PSF_size_m"] # 50µm in [m]
            crop = get_crop(self.psf.shape[0],targetsize_m=targetsize,\
                 pixelsize=self.camera_data["pixelSize"],odd=odd)
        else:
            crop = get_crop(self.psf.shape[0],self.crop_to_size[0])

        if crop is not None:
            self.psf = self.psf[crop,crop]

#EOF
