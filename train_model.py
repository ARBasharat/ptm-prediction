# -*- coding: utf-8 -*-
"""
Abdul Rehman Basharat and Sayali Tailware
Cloud Computing Project
Prediction of Post-Translational modification sites
"""

import os
import keras
import numpy as np
from functools import partial
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def vgg_3(in_shape=(21, 25), n_classes=1, n_stages_per_blocks=[2, 2, 2, 2, 2]):
  in_layer = keras.layers.Input(in_shape)
  block1 = _block(in_layer, 64, n_stages_per_blocks[0])
  pool1 = keras.layers.MaxPool1D()(block1)
  #drop1 = keras.layers.Dropout(0.1)(pool1)
  block2 = _block(pool1, 128, n_stages_per_blocks[1])
  pool2 = keras.layers.MaxPool1D()(block2)
  #drop2 = keras.layers.Dropout(0.1)(pool2)
  block3 = _block(pool2, 256, n_stages_per_blocks[2])
  pool3 = keras.layers.MaxPool1D()(block3)
  #drop3 = keras.layers.Dropout(0.1)(pool3)
  block4 = _block(pool3, 512, n_stages_per_blocks[3])
  pool4 = keras.layers.MaxPool1D()(block4)
  drop4 = keras.layers.Dropout(0.25)(pool4)
  flattened = keras.layers.GlobalAvgPool1D()(drop4)
  dense1 = keras.layers.Dense(2048, activation='relu')(flattened)
  dense2 = keras.layers.Dense(1024, activation='relu')(dense1)
  preds = keras.layers.Dense(n_classes, activation='sigmoid')(dense2)
  model = keras.models.Model(in_layer, preds)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

conv3 = partial(keras.layers.Conv1D, kernel_size=3, strides=1, padding='same', activation='relu')
def _block(in_tensor, filters, n_convs):
  conv_block = in_tensor
  for _ in range(n_convs):
    conv_block = conv3(filters=filters)(conv_block)
  return conv_block

def print_training_history(history):
  training_file_name = os.path.join("TrainingHistory.txt")
  f = open(training_file_name, "w")
  f.writelines("Training Data History") # Write a string to a file
  f.writelines("\nTraining_Accuracy :" + str(history.history['acc']) + " \nValidation_Accuracy :" + str(history.history['val_acc']) ) # Write a string to a file
  f.writelines("\nTraining_Loss :" + str(history.history['loss']) + " \nValidation_Loss :" + str(history.history['val_loss']) ) # Write a string to a file
  f.close()

def plot_training_graphs(history):
  _plot_loss_graph(history)
  _plot_accuracy_graph(history)

def _plot_loss_graph(history):
  graph_file_name = os.path.join(os.getcwd(), "ModelLoss.png")
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')  
  plt.ylabel('loss')  
  plt.xlabel('epoch')  
  plt.legend(['train', 'test'], loc='upper left')  
  plt.savefig(graph_file_name, dpi=250)
  plt.close()

def _plot_accuracy_graph(history):
  graph_file_name = os.path.join(os.getcwd(), "ModelAccuracy.png")   
  plt.figure()
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(graph_file_name, dpi=250)
  plt.close()

## Load Train Data
train_data = np.load(os.path.join(os.getcwd(), "train_data.npz"))
x_train = train_data['arr_0']
y_train = train_data['arr_1']

## Load Validation Data
val_data = np.load(os.path.join(os.getcwd(), "val_data.npz"))
x_val = val_data['arr_0']
y_val = val_data['arr_1']

## Train Model and save the best version
model = vgg_3()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), "model.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, callbacks=[early_stopping, checkpoint], validation_data=(x_val, y_val))
print_training_history(history)
plot_training_graphs(history)