import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

data_dir = "../Data_sets/flower_photos"
# data_dir = keras.utils.get_file('flower_photos', origin = dataset_dir, untar = True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Dataset has :',image_count,' files')

batch_size = 32
img_hight = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_hight,img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_hight,img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("class labels : ",class_names)
