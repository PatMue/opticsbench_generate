# pm23 , 29.01.2023

import numpy as np
import matplotlib.pyplot as plt 
import logging 

logger = logging.getLogger(__name__)

def get_next_odd_integer(x:float):
    """!"""
    if round(x) % 2: # e.g. 24.51... 25.50
        return int(round(x))
    else:
        if np.floor(x) % 2: # 25.51 ... 25.99 
            return int(np.floor(x))
        elif np.ceil(x) % 2: # 24.000001 ... 25
            return int(np.ceil(x))
        return x + 1   # int(24)
        
        
def get_crop(a,b=None,targetsize_m=108e-6,pixelsize=4.46e-6,odd=True):
    """!
	a: current size  (1d)
	b: target size  (1d)
    # A: even, B: even  # A: even, B: odd 
    # A: odd, B: even   # A: odd, B: odd 
    """
    if b is None: 
        if odd:
            b = get_next_odd_integer(targetsize_m / pixelsize)
        else:
            b = int(round(targetsize_m / pixelsize))	
    else:
        if odd:
            b = get_next_odd_integer(b)
    ctr = int(round(a / 2)) + 1 # update -1
    crop = [b//2 +1, b//2] if b % 2 else [b//2,b//2]    
    logger.debug(f'get_crop: a: {a} --> b: {b}')
    return slice(ctr-crop[0],ctr+crop[1],1)


def soft_plot_close(close:bool=True,wait:float=False):
    plt.show(block=False)
    if wait:
        plt.pause(wait)
        __ = input("\npress [return] to close plot")
    else:
        __ = input("Continue...")
    if not close:
        plt.clf()
    else:
        plt.close()


def test(sz=104):
    """"!
    verify that center is still at the right place
    verify target shape 
    """
    import matplotlib.pyplot as plt 
    targetsize = 25
    for sz in [104,105]:
        dummy = np.zeros([sz,sz])
        dummy[sz//2,sz//2] = 1.0
        s = get_crop(dummy.shape[0],b=targetsize)
        cropped = dummy[s,s]
        assert targetsize == cropped.shape[0], f"Size mismatch: {targetsize},{cropped.shape[0]}"
        fig,ax = plt.subplots(1,2)
        print(f"center: {cropped.shape[0]/2} --> {cropped.shape[0]//2}")
        ax[0].imshow(dummy)
        ax[1].imshow(cropped)
        plt.suptitle(f"before: {dummy.shape}, after: {cropped.shape}")
        plt.show(block=False)
        input()
        plt.close()


if __name__ == "__main__":	
   test()