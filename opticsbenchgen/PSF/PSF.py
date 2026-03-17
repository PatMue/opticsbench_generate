"""@package docstring
Module PSF -- generate psf from zernike polynomials. 
"""
import logging
import copy

import pyfftw
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage
from PIL import Image as PIL_IMG


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


def old_version(func):
    def fcn(*args, **kwargs):
        logger.info(f"(deprecation warning): The function {func.__name__} is deprecated." +\
            " See __doc__ for further infos. (experimental)")
        return func(*args, **kwargs)
    return fcn


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
        camera = self._wrap_camera_names(camera) if as_api else camera
        self._update_camera(camera)
        self.get_physical_grid_from(physical_dim)
        self.get_pupil_function()
        self.__summarize__()


    def _wrap_camera_names(self,camera:dict):
        """!"""
        try:
            cam = copy.deepcopy(camera)
            return {\
                "sampling":cam["exit_pupil_sampling_grid_size"],\
                "xp_diameter":cam["exit_pupil_Diameter"],\
                "xp_distance":cam["focus_length"],\
                "wave":cam["wavelength_used"]}
        except KeyError as err:
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
            if cam.__contains__(key):
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
        attr = [attr for attr in self._get_options() if physical_dim.__contains__(attr)]
        self.physical = bool(attr.__len__())
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
                raise(ValueError,f"attribute does not exist: {attr}")

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


    def __summarize__(self):
        """!"""
        if self.physical:
            fcn = logger.debug if not self.verbose else print
            fcn("\n...........")
            fcn("The following setup is used:")
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


    def _get_aggregated_wavefront_(self,coeff:np.ndarray,n:np.ndarray=None,m:np.ndarray=None):
        """!
        Compute the wavefront using Zernike polynomials:
            @param[in] coeff: [np.ndarray]  the coefficient values
            @param[in] n: [np.ndarray] the polynomial index
            @param[in] m: [np.ndarray] the polynomial index
        """
        self.W = np.reshape(np.squeeze(np.dot(coeff.transpose(),self.Z))[::-1],self.W.shape)#.transpose())[::-1]
        self.W = self.pp * self.W  #+ np.mean(self.W[self.pp_mask])
        logger.debug("np.mean W: {}".format(np.mean(self.W[self.pp_mask]))) # 1.56..


    def __create_imager__(self):
        """!
        Create the specified imager.
        """
        pixelSize = self.camera["pixelSize"]
        imsize = self.camera["imsize"]
        sensor_dim = [round(pixelSize*sz,2) for sz in imsize]
        self.imager_center = [imsize[0]//2,imsize[1]//2]

        fcn = logger.info if not self.verbose else print
        fcn("----------------\nphysical grid for imager created")
        fcn("camera_sensor_dim: [height:y,width:x] " + str(sensor_dim*1e3) + " mm")


    def _get_fringe_aggregated_wavefront_without_tilt(self,coeff):
        """!"""
        tilt_xy_removed = copy.deepcopy(coeff[1:3])
        coeff_temp = copy.deepcopy(coeff)
        coeff_temp[1:3] = 0 # set to zero in storage for the copied list
        print(f"Removing tilt x and y: {tilt_xy_removed} . Total of {len(coeff)} coefficients.")
        self._get_aggregated_wavefront_(coeff_temp)
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
    Apply Zernike polynomials to form a PSF , (c) Patrick Müller 2020-2023
    """
    # static class attributes,
    rho = np.array([]) # polar coordinates
    phi = np.array([])
    idx = np.array([])
    pp = np.array([]) # pupil function
    z = np.array([]) # zernike => however, is defined as psf collector obj ->"Z"
    W = np.array([])  # phase error
    # the final PSF:
    PSF = np.array([])
    # additional attributes (pupil and imager definition):
    x = None
    delta_x = None
    df = None
    fx = None
    pp_pupil_size_px = None
    pp_pixel_size = None
    pp_radius = None
    relative_pupil_area = None
    physical_pixel_size = None
    physical_PSF_extent = None
    scaling_factor = None
    imager_resolution = None
    imager_center = None
    crop_to_size = None

    """
    Initialize the PSF simulation based on pupil simulation
    """
    def __init__(self,camera_data:dict=None,interpolation_functions:list=None,\
            balance_coeff_fcns:list=None,scale_by_pixel_size:bool=True,\
            accelerate_with_CUDA:bool=False,j_max=20,use_old_version:bool=False,\
            verbose=False):
        """! Initializes the PSF class.
        @param camera_data: [dict] with all camera definitions
        @param interpolation_functions: [list] with coefficient fcns for interpolated imager grid
        @param balance_coeff_fcns: [list] containing parameter variations
        @param scale_by_pixel_size: [bool]
        @param accelerate_with_CUDA: [bool] whether to use GPU or CPU processing (if available)
        """
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
        self.generate_pupil()
        logger.debug("initialized psf object with pupil {}".format(self.pp.shape))


    def generate_pupil(self,use_old_version=False,circular_pupil_sampling_min=32):
        """!
        generates a physical pupil.
        """
        report_fcn = logger.info if not self.verbose else print
        camera_data = self.camera_data
        if not use_old_version:
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
                    report_fcn(msg)
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
                    report_fcn("Scaling from {} to {}--> {}.".format(pupil.dx,camera_data["pixelSize"],\
                        self.scaling_factor))
                    logger.info("Scaling from {} to {}--> {}.".format(\
                        pupil.dx,camera_data["pixelSize"],self.scaling_factor))
            assert circular_pupil_sampling >= 32, "Increase sampling to allow for processing."
            self.pupilgen = pupil
        else:
            raise NotImplementedError


    @old_version
    def getWavefront(self,coeff:np.ndarray,n:np.ndarray,m:np.ndarray):
        """!
        Compute the wavefront using Zernike polynomials
        @param[in] coeff: [np.ndarray]  the coefficient values
        @param[in] n: [np.ndarray] the polynomial index
        @param[in] m: [np.ndarray] the polynomial index
        """
        self.W = np.zeros(self.pp.shape) # initialize array of size pp.shape --> e.g. 512x512
        self.W[self.pp_mask] = np.squeeze(np.dot(coeff.transpose(),\
            [ZernPoly.zernike_fun_optimizer(int(n[k]),int(m[k]),\
                self.rho[self.pp_mask],self.phi[self.pp_mask]) for k in range(0,len(n))]).transpose())


    def getWavefront_using_Chongs_method(self,coeff:np.ndarray,n:np.ndarray,m:np.ndarray):
        """!
        Compute the wavefront using Zernike polynomials and Chong et al. method (acceleration)
        @param[in] coeff: [np.ndarray]  the coefficient values
        @param[in] n: [np.ndarray] the polynomial index
        @param[in] m: [np.ndarray] the polynomial index
        """
        self.W = np.zeros(self.pp.shape) # initialize array of size pp.shape -->
        self.W[self.pp_mask] = np.squeeze(np.dot(coeff.transpose(),\
            ZernPoly.zernike_fun_using_chong_et_al_method(n,m,\
            self.rho[self.pp_mask],self.phi[self.pp_mask])).transpose())


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


    def getWavefront_using_precomputed_Z_batch(self,coeff_batch:np.ndarray):
        """!
        Use precomputed single Zernike Polynomials to acquire the wavefront
        @param[in] coeff: [np.ndarray] <batch_size,num_coeffs>
        Z has shape: <num_coeffs,prod(pp.shape)>
        @param[out] W: [np.ndarray]  <batch_size,pp.shape[0],pp.shape[1]>
        """
        logger.debug("this function can only be used if all wavefronts have the same" +\
            "(number) of polynomials required <-> n_list_ij == n_list for m respectively)")
        logger.debug("shape of coeff_batch: {}, Z: {}, pp,W: {}".format(\
            coeff_batch.shape,np.shape(self.Z),self.pp.shape))
        return np.reshape(np.dot(coeff_batch,self.Z),(coeff_batch.shape[0],*self.pp.shape))


    def getWavefront_unoptimized(self,coeff:np.ndarray,n:np.ndarray,m:np.ndarray):
        """!
        Compute the wavefront using Zernike polynomials (no optimization)
        @param[in] coeff: [np.ndarray]  the coefficient values
        @param[in] n: [np.ndarray] the polynomial index
        @param[in] m: [np.ndarray] the polynomial index
        """
        print("getWavefront: Unoptimized function called - n-dimensional dot product \
            and  is much faster than the below implementation")
        self.W = np.zeros(self.pp.shape)
        for k,nn in enumerate(n):
            self.z = ZernPoly.zernike_fun(int(nn),int(m[k]),self.rho[self.pp_mask],self.phi[self.pp_mask])
            self.W[self.pp_mask] += float(coeff[k])*self.z


    def get_coefficients_for_current_position(self,x:float,y:float,\
            interpolation_mode="bilinear",verbose=False,balance_coeff_fcns=None,\
            use_zemax_export=True):
        """!
        @param[in] x: x_position in px [col]
        @param[in] y: y_position in px [row]
        @param[in] verbose: False, True: print the current coefficients from the interpolation
        @param[out] coeff: array with all coefficients for the current position to form \
            a phase error function, ordering as set in Dataflow
        """
        if balance_coeff_fcns is not None:
            if "bilinear" in interpolation_mode:
                coeff = np.array([f.__call__(y,x) for f in self.interpolation_functions])
            elif "linear_1D" in interpolation_mode:
                coeff =  np.array([f(np.sqrt((y-self.imager_center[0])**2 + \
                    (x-self.imager_center[1])**2)) for f in self.interpolation_functions])
            else:
                coeff =  np.array([f.__call__(y,x) for f in self.interpolation_functions])

            if use_zemax_export:
                factors = balance_coeff_fcns
                logger.debug(f"coeff before: {coeff[:5]}")
                coeff =  np.multiply(coeff,factors)
                logger.debug(f"coeff after: {coeff[:5]}")
                return coeff
            else:
                logger.debug("coeff before: " + str(coeff[1]))
                coeff = np.multiply(coeff,[balance_coeff_fcns[j](\
                    get_normed_coordinate(y,self.camera_data["imager_center"][0])) if (j \
                        < len(balance_coeff_fcns)) else 1 for j \
                        in range(0,len(self.interpolation_functions))])  # weight tilt only
                logger.debug("coeff after: " + str(coeff[1])) #pass # coeff[1] = -40#-30#-30 # in px
                return coeff

        if "bilinear" in interpolation_mode:
            return np.array([f.__call__(y,x) for f in self.interpolation_functions])
        if "linear_1D" in interpolation_mode:
            return  np.array([f(np.sqrt((y-self.imager_center[0])**2+(x\
                -self.imager_center[1])**2)) for f in self.interpolation_functions])
        return  np.array([f.__call__(y,x) for f in self.interpolation_functions])


    def getPSF(self,showWarning=True):
        """!
        Deprecated.
        """
        if showWarning:
            print('deprecated: call getPSF_using_pyfftw() is much faster')
        P = np.multiply(self.pp,np.exp(-1j*(+2)*np.pi*self.W))
        self.PSF = np.abs(np.fft.fftshift(np.fft.fft2(P),axes=None)*self.df**2)**2
        self.PSF = self.PSF/self.PSF.sum()


    def getPSF_using_pyfftw(self,pyfftwObj):
        """
        Get the psf using pyfftw acceleration.
        @[in] pyfftwObj: Initialized in Dataflow class once for use in for loop to reduce overhead
        """
        # https://github.com/brandondube/prysm/blob/master/prysm/propagation.py
        logger.debug("W: {}, pupil: {}, df: {}".format(self.W.shape,self.pp.shape,self.df))
        self.PSF = np.abs(np.fft.fftshift(pyfftwObj(np.multiply(\
            self.pp,np.exp(-1j*(+2*np.pi)*self.W))),axes=None)*self.df**2)**2
        self.PSF = self.PSF / self.PSF.sum() # 0...1


    def prepare_PSF_collector(obj,specs:dict,n:np.ndarray=None,m:np.ndarray=None,\
            interpolation_functions:list=None,balance_coeff_fcns:list=None,verbose=False):
        """!
        ###############################################################
        PSF Simulation with pupil functions , v. 0.10 (c) Patrick Müller
        ###############################################################
        @param[in] obj: Dataflow object
        @param[in] specs: dictionary containing specs from Dataflow processing function
        @param[in] ff: interpolation functions created by \
            Dataflow.create_coefficient_map_with_selected_ordering
        @param[in] verbose=False: True -> show some informations about the optics and data
        @param[out] zern: PSF object using ZernPoly
        """
        PSF_collector = PSF(specs["camera_data"],interpolation_functions=\
            interpolation_functions,balance_coeff_fcns=balance_coeff_fcns,\
            accelerate_with_CUDA=specs["processing"]["accelerate_with_CUDA"],\
            j_max=specs["PSF_generation"]["j_max"],verbose=verbose)
        PSF_collector.create_imager()
        PSF_collector.crop_to_size = obj.crop_to_size
        if specs["PSF_generation"]["reference_pp_diameter_px"]:
            if specs["PSF_generation"]["use_sqrt"]:
                PSF_collector.relative_pupil_area = \
                    np.sqrt((specs["PSF_generation"]["reference_pp_diameter_px"]**2) \
                    /(PSF_collector.simulated_pp_diameter_px**2))
            else:
                PSF_collector.relative_pupil_area = \
                    (specs["PSF_generation"]["reference_pp_diameter_px"]**2) \
                    /(PSF_collector.simulated_pp_diameter_px**2)
        else:
            PSF_collector.relative_pupil_area = [] # do nothing

        if not specs["PSF_generation"]["use_zemax_export"]:
            PSF_collector.Z = ZernPoly.zernike_fun_using_chong_et_al_method(n,
                m,
                PSF_collector.rho[PSF_collector.pp_mask],
                PSF_collector.phi[PSF_collector.pp_mask],
                pp_factor = PSF_collector.camera_data["pp_factor"],
                pp_relative_area = [])#PSF_collector.relative_pupil_area) # relative area
        else:
            logger.info("Precomputed zernike polynomials of shape: {}".format(\
                np.shape(PSF_collector.Z)))

        logger.debug("Transition zone for z0 is at about: " + \
            str(round(specs["camera_data"]["exit_pupil_Diameter"]**2 / (500*1e-9),4)) + \
                " meters")
        return PSF_collector


    def get_psf_from_sample_coeffs(PSF_collector,specs:dict,pyfftwObj,coeffs:np.ndarray,\
            binning=False,rotate=False):
        """!"""
        PSF_collector.getWavefront_using_precomputed_Z(coeffs)
        PSF_collector.getPSF_using_pyfftw(pyfftwObj)
        if rotate and "linear_1D" in specs["PSF_generation"]["interpolation_mode"]:
            PSF_collector.rotate_using_ndimage(x,y) #(x[1],y[1])
        if binning:
            PSF_collector.binning(binFactor = specs["PSF_output"]["binFactor"])
        return PSF_collector.PSF


    def get_current_PSF(PSF_collector,specs:dict,pyfftwObj,x:float,\
            y:float,n:np.ndarray,m:np.ndarray):
        """
        @param[in] x: x_position in image for meta data access
        @param[in] y: y_position in image for meta data access
        @param[in] coeff: all coefficients to generate the current
        @param[in] specs:
        """
        if specs["processing"]["use_chongs_method"]:
            PSF_collector.getWavefront_using_Chongs_method(\
                PSF_collector.get_coefficients_for_current_position(x,y,\
                interpolation_mode=specs["PSF_generation"]["interpolation_mode"],\
                verbose=False,use_zemax_export=specs["PSF_generation"]["use_zemax_export"]),n,m) # no weight => A_coeff = 1,
                #write to PSF_collector.W without explicitly overwriting the object
        elif specs["processing"]["use_chongs_method"] is None:
            PSF_collector.getWavefront_using_precomputed_Z(\
                PSF_collector.get_coefficients_for_current_position(x,y,\
                interpolation_mode=specs["PSF_generation"]["interpolation_mode"],\
                use_zemax_export=specs["PSF_generation"]["use_zemax_export"]))
        else:
            PSF_collector.getWavefront(PSF_collector.get_coefficients_for_current_position(\
                x,y,interpolation_mode=specs["PSF_generation"]["interpolation_mode"],\
                verbose=False,use_zemax_export=specs["PSF_generation"]["use_zemax_export"]),n,m)
        PSF_collector.getPSF_using_pyfftw(pyfftwObj)

        if specs["PSF_generation"]["interpolation_mode"].get("linear_1D",False):
            PSF_collector.rotate_using_PIL(x,y)
        PSF_collector.binning(binFactor = specs["PSF_output"]["binFactor"])
        
        return PSF_collector.PSF / PSF_collector.PSF.sum()

 
    def rotate_using_ndimage(self,x,y,verbose=False):
        """!
        Rotate PSF using scipy.ndimage.rotate (for consistency checks)
        """
        angle = np.rad2deg(-np.arctan2((y-self.imager_center[0]),(x-self.imager_center[1]))) -90
        # fixed to +angle --> using validation proof
        if angle >= 180:
            angle = 180 - angle
        self.PSF = ndimage.rotate(self.PSF,
                            angle,
                            axes=(1, 0),
                            reshape=False,
                            output=None,
                            order=3,
                            mode='constant',
                            cval=0.0,
                            prefilter=True)
        return self.PSF/np.max(self.PSF) # 0...1


    def binning(self,verbose = False, binFactor:list=None):
        """
        @param[in] binFactor:   Target size to resample the PSF to.
                                None: no binning,
                                Empty list []: automatic (use physical representation) -> default
                                [0.5]: user scaling factor
                                
                                (order = 1 as experiments suggested on 17.02.2023)
        """
        if binFactor is not False:
            grid_mode = True if not self.PSF.shape[0] % 2 else False
            logger.debug("Note that with the current implementation not all PSF sizes \
                are available --> e.g. [32,32] but not [30,30]")
            logger.debug("self.PSF.shape before binning: " + str(self.PSF.shape))

            self.PSF = ndimage.zoom(self.PSF, self.scaling_factor, output=None, \
                 order=3, mode='constant', cval=0.0, prefilter=True,grid_mode=grid_mode)
                                         
            logger.debug("Now the PSF has approximately pixel size of the target imager!")
            logger.debug("PSF [px]: " + str(self.PSF.shape))
            logger.debug("crop the PSF size from " + \
                str(round(self.PSF.shape[0]*self.camera_data["pixelSize"]*1e6,3)) +\
                "µm to 50µm maximum: ")
            logger.debug("Pupil full grid size: " + str(self.pp.shape[0]))
            logger.debug("PSF shape after binning: " + str(self.PSF.shape))
            if verbose:
                self.plot_physical_PSF(wait_for_user_input=True)


    def crop(self,verbose=False,odd=True):
        """ crop the psf to physical size / target size."""
        if self.crop_to_size is None:
            targetsize = self.camera_data["maximum_PSF_size_m"] # 50µm in [m]
            logger.debug("target size: {}µm, actual: {}µm".format(1e6*targetsize,\
                self.PSF.shape[0]*self.camera_data["pixelSize"]*1e6))                
            crop = get_crop(self.PSF.shape[0],targetsize_m=targetsize,\
                 pixelsize=self.camera_data["pixelSize"],odd=odd)
        else:
            crop = get_crop(self.PSF.shape[0],self.crop_to_size[0])
            logger.debug("crop_to_size: {}".format(self.crop_to_size))

        if crop is not None: # otherwise it would be no crop
            self.PSF = self.PSF[crop,crop]
            logger.debug(f"cropped (odd: {odd}) to {self.PSF.shape} using slice: {crop}")

        logger.debug("PSF shape after binning and cropping: " + str(self.PSF.shape))
        logger.debug("PSF size is now: {} [px] and {}µm".format(self.PSF.shape, \
            round(self.PSF.shape[0]*self.camera_data["pixelSize"]*1e6,4)))
        logger.debug("self.physical_pixel_size [µm]" + str(round(self.physical_pixel_size*1e6,4)))
        logger.debug("camera_data[pixelSize]: " + str(round(self.camera_data["pixelSize"]*1e6,4)) + "µm")
        

#EOF
