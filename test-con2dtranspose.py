import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np

# x=tf.convert_to_tensor( [[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]])
x = tf.reshape(( [[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]],[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]]),(2.,3.,3.))
print(x)
model = tf.keras.Sequential()
model.add(
    layers.Conv2DTranspose(1,(5,5),strides = (2,2), padding = 'same', use_bias = False ,kernel_initializer='ones'))

print(model(x,training=False))
