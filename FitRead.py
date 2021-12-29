import numpy as np
import matplotlib.pyplot as plt
import glob, os
from scipy import ndimage
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits


def plot_picture(image, inf = 0, sup = 1, step = 1, colours = 'hot' ):
    ''' It plots figures '''
    plt.style.use(astropy_mpl_style) #telling astropy about using plt
    plt.figure()
    plt.grid(b=None)
    #plt.imshow(image,  extent=[-size, size, -size, size], cmap = colours)
    plt.plot()
    plt.show()


filenames = glob.glob('/home/conpucter/Desktop/DataSRH/2016/18/*')
filenames = sorted(filenames, key=os.path.basename)

for filename in filenames[0:1]:
	image_file = fits.open(filename)
	print(image_file[1].header)
	print(image_file.info())
	#plot_picture(data)

