import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

dataset_dir = "../data_sets/flower_photos.tgz"
data_dir = keras.utils.get_file('flower_photos', origin = dataset_dir, untar = True)