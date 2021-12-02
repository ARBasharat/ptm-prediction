# -*- coding: utf-8 -*-
"""
Abdul Rehman Basharat and Sayali Tailware
Cloud Computing Project
Prediction of Post-Translational modification sites
"""
import os
import keras.models
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def generate_roc_curve(predicted_label_probabilities, assigned_label, file_name="ROC_Curve.png"):
  auc = roc_auc_score(assigned_label, predicted_label_probabilities)
  print('EnvCNN_AUC: %.3f' % auc)
  fpr, tpr, thresholds = roc_curve(assigned_label, predicted_label_probabilities)
  plot_roc_test(fpr, tpr, file_name)

def plot_roc_test(fpr, tpr, roc_file_name):
  plt.figure()
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr, marker='.')
  plt.legend(['Reference', 'EnvCNN'], loc='lower right')
  plt.savefig(roc_file_name, dpi=250)
  plt.close()

if __name__ == "__main__":
  ## Load Train Data
  test_data = np.load(os.path.join(os.getcwd(), "test_data.npz"))
  x_test = test_data['arr_0']
  y_test = test_data['arr_1']
  
  model = keras.models.load_model(os.path.join(os.getcwd(), "model.h5"))
  model.compile(loss = "binary_crossentropy", metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=5e-05))
  
  history = model.evaluate(x_test, y_test, verbose=1)
  print(history)
  predictions = model.predict(x_test, verbose=1)
  generate_roc_curve(predictions, y_test)
