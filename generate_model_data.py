# -*- coding: utf-8 -*-
"""
Abdul Rehman Basharat and Sayali Tailware
Cloud Computing Project
Prediction of Post-Translational modification sites
"""

import os
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from pyspark import SparkContext

class ProteinDatabase:
  def __init__(self, protein_dictionary):
    self.protein_dictionary = protein_dictionary

  @classmethod
  def read_file(cls, database_file):
    protein_dictionary = ProteinDatabase._read_database(database_file)
    return cls(protein_dictionary)

  def __getitem__(self, key):
    return self.protein_dictionary[key]

  def _get_key_list(self):
    return list(self.protein_dictionary.keys())

  @staticmethod
  def _read_database(database_file):
    fasta = []
    protein_sequences = []
    protein_headers = []
    with open(database_file) as input_file:
      for line in input_file:
        line = line.strip()
        if not line:
          continue
        if line.startswith(">"):
          active_sequence_name = line[1:]
          if active_sequence_name not in fasta:
            protein_sequences.append(''.join(fasta))
            protein_id = active_sequence_name
            protein_header = protein_id.split('|')
            protein_header = protein_header[1]
            protein_headers.append(protein_header)
            fasta = []
          continue
        sequence = line
        fasta.append(sequence)
      # Flush the last fasta block to the ProteinSequences list
      if fasta:
        protein_sequences.append(''.join(fasta))
        del protein_sequences[0]
    return dict(zip(protein_headers, protein_sequences))

def to_low(seq):
  y = [x.start() for x in re.finditer("Y", seq)]
  s = [x.start() for x in re.finditer("S", seq)]
  t = [x.start() for x in re.finditer("T", seq)]
  start_locations = y + s + t
  sites = []
  for start_idx in start_locations:
    site = ""
    for i in range(-10, 11, 1):
      idx = start_idx + i
      if idx < 0 or idx > len(seq)-1:
        site = site + "-"
      else:
        site = site + seq[idx]
    sites.append(site)
  return sites

def _get_training_matrix(short_seqs):
  window_size = 21
  letterDict = {}
  letterDict["A"] = 0
  letterDict["C"] = 1
  letterDict["D"] = 2
  letterDict["E"] = 3
  letterDict["F"] = 4
  letterDict["G"] = 5
  letterDict["H"] = 6
  letterDict["I"] = 7
  letterDict["K"] = 8
  letterDict["L"] = 9
  letterDict["M"] = 10
  letterDict["N"] = 11
  letterDict["P"] = 12
  letterDict["Q"] = 13
  letterDict["R"] = 14
  letterDict["S"] = 15
  letterDict["T"] = 16
  letterDict["V"] = 17
  letterDict["W"] = 18
  letterDict["Y"] = 19
  letterDict["B"] = 20
  letterDict["Z"] = 21
  letterDict["U"] = 22
  letterDict["X"] = 23
  letterDict["-"] = 24
  ONE_HOT_SIZE = len(letterDict)
  Matr = np.zeros((len(short_seqs), window_size, ONE_HOT_SIZE))
  samplenumber = 0
  for seq in short_seqs:
    AANo = 0
    for AA in seq:
      index = letterDict[AA]
      Matr[samplenumber][AANo][index] = 1
      AANo = AANo+1
    samplenumber = samplenumber + 1
  return Matr

def _read_protein_ids(mod_file):
  ## Read file
  f = open(mod_file, 'r')
  all_lines = f.readlines()
  all_lines = [x.strip() for x in all_lines] 
  f.close()
  ## Process Data
  protein_id = []
  for line in all_lines:
    line_components = line.split('\t')
    protein_id.append(line_components[1])
  return protein_id
  
def _read_mod_file(mod_file):
  ## Read file
  f = open(mod_file, 'r')
  all_lines = f.readlines()
  all_lines = [x.strip() for x in all_lines] 
  f.close()
  ## Process Data
  binding_seq = []
  protein_id = []
  mod_location = []
  for line in all_lines:
    line_components = line.split('\t')
    protein_id.append(line_components[1])
    mod_location.append(line_components[1])
    mod_site = line_components[len(line_components) -1][10]
    if mod_site == 'Y' or mod_site == 'S' or mod_site == 'T':
      binding_seq .append(line_components[len(line_components) -1])
  return binding_seq

if __name__ == "__main__":
  sc = SparkContext()
  ## Reading modification file and Generate Positive Data
  mod_file = "Phosphorylation.txt"
  pos_mod_sites = _read_mod_file(mod_file)
  mod_proteins = _read_protein_ids(mod_file)

  ## Reading Protein Database
  database_file = "UP000005640.fasta"
  prot_db = ProteinDatabase.read_file(database_file)
  protein_id_keys = prot_db._get_key_list()

  ## Generate Negative training Data
  training_ids = list(set(mod_proteins) & set(protein_id_keys))
  rdd = sc.parallelize(training_ids)
  sitesRdd = rdd.map(lambda x: prot_db[x]).flatMap(lambda x: to_low(x))
  sites = sitesRdd.collect()

  ## Shortlist Negative Training Data
  neg_elems = list(set(sites) - set(pos_mod_sites))
  neg_mod_sites = random.sample(neg_elems, len(pos_mod_sites))

  ## Generate Training Data
  mod_data = pos_mod_sites + neg_mod_sites
  mod_labels = np.array([1] * len(pos_mod_sites) + [0] * len(neg_mod_sites))
  mod_matrix = _get_training_matrix(mod_data)

  ## Split and Shuffle Data
  train_ratio = 0.75
  validation_ratio = 0.10
  test_ratio = 0.15

  # train is 75%, test is 15% and validation is 10% of the initial data set
  x_train, x_test, y_train, y_test = train_test_split(mod_matrix, mod_labels, test_size=1 - train_ratio)
  x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

  ## Save npy files -- Train, Test, and Validation
  np.savez(os.path.join(os.getcwd(), "train_data"),  x_train, y_train)
  np.savez(os.path.join(os.getcwd(), "test_data"),  x_test, y_test)
  np.savez(os.path.join(os.getcwd(), "val_data"),  x_val, y_val)
