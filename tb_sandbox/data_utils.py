import math
import sys

import dask.array as da
import numpy as np
from skimage.morphology import binary_erosion, label, remove_small_objects


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def size(object):
    """
    Lazy lazy way to return the size of a data object
    """
    return convert_size(sys.getsizeof(object))


def binarise_masks(masks):
    """
    Quick function to change instance segmentation map to semantic segmentation
    """

    new_masks = list()
    for frame in masks:
        frame = remove_small_objects(frame.compute(astype='i2'),
                                     min_size=1000)
        labelled = label(frame)
        new_mask = np.zeros(labelled.shape)
        for segment_ID in range(1, np.max(labelled) + 1):
            segment = binary_erosion(labelled == segment_ID)
            new_mask[segment] = 1
            # set background to zero
            new_mask[labelled == 0] = 0
            new_mask = new_mask.astype('i2')
        new_masks.append(new_mask)
    new_masks = da.stack(new_masks, axis=0)

    return new_masks


def track_to_df(track):
    """
    Quick hack to return a single track as a dataframe for output into excel
    """
    import pandas as pd

    return pd.DataFrame(track.to_dict(), columns=list(track.to_dict().keys()))
