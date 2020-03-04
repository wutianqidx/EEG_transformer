import numpy as np
import torch
import pandas as pd

class EegDataset:
    def __init__(self, pkl_file):
        self.data = pd.read_pickle(pkl_file)

    def __getitem__(self, idx):

        return input,output

class PackCollate:
    def __call__(self, batch):

        return input_batch,output_batch
