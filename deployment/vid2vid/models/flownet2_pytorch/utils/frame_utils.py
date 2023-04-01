import numpy as np
from os.path import *
from scipy.misc import imread
from . import flow_utils 

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext in ['.png', '.jpeg', '.ppm', '.jpg']:
        im = imread(file_name)
        return im[:,:,:3] if im.shape[2] > 3 else im
    elif ext in ['.bin', '.raw']:
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []
