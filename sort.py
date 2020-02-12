import tensorflow as tf
import os
import glob
from pathlib import Path
from shutil import copy
import numpy as np

model = tf.keras.models.load_model("./best_model.hdf5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.layers[0].input_shape)
files = os.listdir('/var/www/wyro/html_old/dataset/all/')

files = files[1:]
files = files[:-1]
maleDestination = '/var/www/wyro/html_old/dataset/male/'
femaleDestination = '/var/www/wyro/html_old/dataset/female/'

for x, y in enumerate(files):
    d = tf.keras.preprocessing.image.load_img('/var/www/wyro/html_old/dataset/all/'+y,
                                              color_mode='grayscale',
                                              target_size=(200, 200))
    d = np.asarray(d)
    d = d.reshape(1, 200, 200, 1)

    prediction = model.predict(d, steps=1)
    src = '/var/www/wyro/html_old/dataset/all/' + y
    if np.round(prediction) == 0:
        copy(src, maleDestination + y)
    elif np.round(prediction) == 1:
        copy(src, femaleDestination + y)

