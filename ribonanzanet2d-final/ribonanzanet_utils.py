import pandas as pd
import torch
import numpy as np
import random
import yaml
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from Network import RibonanzaNet
import torch.nn as nn
from mamba_ssm import Mamba2

# Set seed for reproducibility
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Configuration class and functions
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

# Dataset classes
class RNA_test_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = [self.tokens[nt] for nt in self.data.loc[idx, 'sequence']]
        sequence = torch.tensor(np.array(sequence))
        return {'sequence': sequence}

class RNA_Dataset(Dataset):
    def __init__(self, data, length=68, max_len=130):
        self.data = data
        self.length = length
        self.max_len = max_len
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}
        self.label_names = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = [self.tokens[nt] for nt in self.data.loc[idx, 'sequence']]
        sequence_len = len(sequence)
        sequence = torch.tensor(np.array(sequence))
        if len(sequence) < self.max_len:
            sequence = torch.cat([sequence, torch.ones(self.max_len - len(sequence)).long() * 4])
        
        # Function to pad tensors to the same size
        def pad_tensor(tensor, target_length):
            padding_length = target_length - tensor.size(0)
            if padding_length > 0:
                return torch.cat([tensor, torch.zeros(padding_length)], dim=0)
            else:
                return tensor[:target_length]

        # Extract tensors from the DataFrame
        tensors = [torch.tensor(self.data.loc[idx, l]) for l in self.label_names]

        # Pad tensors to the target length
        padded_tensors = [pad_tensor(tensor, self.length) for tensor in tensors]

        # Stack the padded tensors
        labels = torch.stack(padded_tensors, dim=-1)
        print(labels.shape)
        print(sequence.shape)
        if len(labels) > self.length:
            labels = labels[:self.length]
        else:
            labels = torch.cat([labels, torch.zeros(self.length - len(labels), 5)])
        if sequence_len > self.length:
            sequence_len = torch.tensor(self.length)
        return {'sequence': sequence, 'labels': labels, 'length': sequence_len}

# Loss function
def MCRMAE(y_pred, y_true):
    colwise_mae = torch.mean(torch.abs(y_true - y_pred), dim=0)
    return torch.mean(colwise_mae)

def slice_outputs(output, labels, seq_length):
    sliced_outputs = []
    sliced_labels = []
    for i in range(output.size(0)):
        sliced_output = output[i, :seq_length[i], :]
        sliced_label = labels[i, :seq_length[i], :]
        sliced_outputs.append(sliced_output)
        sliced_labels.append(sliced_label)

    final_output = torch.cat(sliced_outputs, dim=0)
    final_labels = torch.cat(sliced_labels, dim=0)
    return final_output, final_labels

def update_pseudo_labels(data):
    pseudo_cols = ['pseudo_reactivity', 'pseudo_deg_Mg_pH10', 'pseudo_deg_pH10', 'pseudo_deg_Mg_50C', 'pseudo_deg_50C']
    real_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    data[real_cols] = data[pseudo_cols].copy()
    return data

# Data loading and processing functions
def load_data(train_pseudo_path, test_pseudo_107_path, test_pseudo_130_path, noisy_threshold):
    data = pd.read_json(train_pseudo_path, lines=True).reset_index(drop=True)
    data_noisy = data[data['signal_to_noise'] <= noisy_threshold].reset_index(drop=True)
    data = data[data['signal_to_noise'] > noisy_threshold].reset_index(drop=True)
    test107 = pd.read_json(test_pseudo_107_path, lines=True).reset_index(drop=True)
    test130 = pd.read_json(test_pseudo_130_path, lines=True).reset_index(drop=True)
    return data, data_noisy, test107, test130

def split_data(data):
    data['length'] = data['sequence'].apply(len)
    kf = StratifiedKFold(n_splits=10, random_state=2020, shuffle=True)
    for train_index, val_index in kf.split(data, data['SN_filter']):
        break
    train_split = data.loc[train_index].reset_index(drop=True)
    val_split = data.loc[val_index].reset_index(drop=True)
    return train_split, val_split

def augment_real_with_pseudo(train_split):
    for col in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']:
        train_split[col] = train_split.apply(lambda x: x[col] + x[f'pseudo_{col}'][68:], axis=1)
    return train_split

def prepare_training_data(train_split, data_noisy, test107, test130, sn_threshold):
    train_step3 = pd.concat([train_split, data_noisy, test107, test130], axis=0).reset_index(drop=True)
    highSN = train_step3[train_step3['signal_to_noise'] > sn_threshold].reset_index(drop=True)
    return train_step3, highSN

# Model class
class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, use_mamba_start=False, use_mamba_end=False):
        super(finetuned_RibonanzaNet, self).__init__(config)
        self.use_mamba_start = use_mamba_start
        self.use_mamba_end = use_mamba_end
        # if use_mamba_start:
        #     self.mamba_start = Mamba2(
        #         # This module uses roughly 3 * expand * d_model^2 parameters
        #         d_model=256, # Model dimension d_model
        #         d_state=128,  # SSM state expansion factor, typically 64 or 128
        #         d_conv=4,    # Local convolution width
        #         expand=2,    # Block expansion factor
        #     )
        #     print("mamba is used at the start")
        if use_mamba_end:
            self.mamba_end= Mamba2(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=256, # Model dimension d_model
                d_state=128,  # SSM state expansion factor, typically 64 or 128
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
            print("mamba is used at the end")
        self.decoder = nn.Linear(256, 5)


    def forward(self, src):
        # if self.use_mamba_start:
        #     sequence_features = self.mamba_start(sequence_features)
        
        sequence_features, pairwise_features = self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        if self.use_mamba_end:
            sequence_features = self.mamba_end(sequence_features)
        output = self.decoder(sequence_features)
        return output.squeeze(-1)
