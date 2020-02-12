from tensorflow import keras
import numpy as np
import pandas as pd
import os
import glob
import cv2
files = os.listdir('./')

files = files[1:]
files = files[:-3]

gender = []

pictures = glob.glob('./*.jpg')


img = []

for x, y in enumerate(files):
    d = y.split('_')
    gender.append(d[1])

for i in pictures:
    image = cv2.imread(i)
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img.append(grayimg)

df = pd.DataFrame(columns=['gender', 'img'])

df['gender'] = gender
df['img'] = img

targets = np.asarray(gender)
targets = targets.astype(int)

train = np.asarray(img)
train = train.reshape(train.shape[0], 200, 200, 1)

model = keras.Sequential()

model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', data_format='channels_last', input_shape=(200, 200, 1)))
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', data_format='channels_last'))
model.add(keras.layers.MaxPool2D(pool_size=(4, 4)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', data_format='channels_last'))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', data_format='channels_last'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', data_format='channels_last'))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', data_format='channels_last'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu', data_format='channels_last'))
model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu', data_format='channels_last'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


checkpoint = keras.callbacks.ModelCheckpoint('best_model.hdf5',
                                             monitor='val_acc',
                                             save_best_only=True,
                                             mode='auto')
stop = keras.callbacks.EarlyStopping(monitor='val_acc',
                                     min_delta=0.1,
                                     patience=50,
                                     verbose=1,
                                     mode='auto',
                                     )

callbacks_list = [checkpoint, stop]
model.fit(train,
          targets,
          batch_size=25,
          validation_split=0.2,
          epochs=1000,
          shuffle=True,
          use_multiprocessing=True,
          callbacks=callbacks_list)

