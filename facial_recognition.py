# Importing necessary libraries
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from IPython.display import SVG, Image
# from livelossplot.tf_keras import PlotLossesCallback
print("Tensorflow version: ", tf.__version__)

# Plot sample images
# utils.dataset.fer.plot_examples_images(plt).show()

# Get number of expressions in the training set
for expression in os.listdir("train/"):
    print(str(len(os.listdir("train/" + expression))) + " " + expression + " images")

# Generate training and validation batches
img_size = 48
batch_size = 64

train_datagen = ImageDataGenerator(horizontal_flip=True)
train_generator = train_datagen.flow_from_directory("train/", target_size=(img_size, img_size), color_mode="grayscale", batch_size=batch_size, class_mode='categorical', shuffle=True)

val_datagen = ImageDataGenerator(horizontal_flip=True)
val_generator = val_datagen.flow_from_directory("test/", target_size=(img_size, img_size), color_mode='grayscale', batch_size=batch_size, class_mode='categorical', shuffle=False)

# CNN Model
model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Visualize Model Architecture
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# Image('model.png', width=400, height=200)

# Train and Evaluate Model
epochs = 15
steps_per_epoch = train_generator.n//train_generator.batch_size
val_steps = val_generator.n//val_generator.batch_size

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001, mode='auto')
checkpoint = ModelCheckpoint('model_weights.h5', monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1)
callbacks = [checkpoint, reduce_lr]

hist = model.fit(x=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks)

# Represent Model as JSON
model_json = model.to_json()
with(open('model.json', 'w')) as json_file:
    json_file.write(model_json)
