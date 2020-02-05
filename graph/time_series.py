import json
import numpy as np
from .NpEncoder import NpEncoder

def _flatten_image(image):
    """Converts a 2d array into a 1d array for easier analysis.
    Image is flattened in row order"""
    temp = np.array(image).flatten()
    return temp.reshape((image.size,1))


def histogram_deltas(images, bins=10):
    """ Returns a 2d array such that each row contains histogram values for each
    category.
    Images: list like set of 2d images.
    bins: integer or list of bin boundaries.
        If an integer, creates n categories and spaces them evenly
        If list, the bounds of the bins are the numbers i.e.
        (a,b,c,d) -> (a,b) (b,c) (c,d)
        note number of bins will list size - 1
    Return 2d array size nxm where n is number of bins, and m is number of samples
    """

    data = [np.histogram(_flatten_image(im), bins=bins)[0] for im in images]
    return np.array(data).T


def _bin_list(data):
    """ enumerates a list for each row in the data,
    [bin0, bin1, bin2 ... binn]
    """
    size = len(data)
    return ['bin%d' % n for n in np.arange(size)]

def hist_to_json(data):
    """the JSON mudule doesn't recognize numpy data types so I made my own lmaooooooooo
    data in is expected to be the histogram form from histogram_deltas()
    to data_to_json to convert the raw set of images to json
    """
    keys = _bin_list(data)
    d = {k:v for k,v in zip(keys, data)}
    return json.dumps(d, cls=NpEncoder)

def data_to_json(data, bins=10):
    """converts a set of images into that histogram json thing, look up numpy
    histogram for information on the bis argument
    """
    return hist_to_json(histogram_deltas(data, bins=bins))
