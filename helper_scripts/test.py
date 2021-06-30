# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:38:10 2021

@author: 963675
"""


from matplotlib import pyplot as plt

import numpy as np
import math

def index_marks(nrows, chunk_size):
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)


def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

chunks = split(dfm, 100)
for c in chunks:
    print("Shape: {}; {}".format(c.shape, c.index))