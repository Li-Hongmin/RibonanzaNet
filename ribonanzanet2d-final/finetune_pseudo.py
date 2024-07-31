import argparse
import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from Network import *
import yaml
from sklearn.model_selection import StratifiedKFold
from ranger import Ranger
from tqdm import tqdm

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
    def __init__(self, data, length=68):
        self.data = data
        self.length = length
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}
        self.label_names = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = [self.tokens[nt] for nt in self.data.loc[idx, 'sequence']]
        sequence_len = len(sequence)
        sequence = torch.tensor(np.array(sequence))
        if len(sequence) < self.length:
            sequence = torch.cat([sequence, torch.ones(self.length - len(sequence)).long() * 4])
        
        labels = torch.tensor(np.stack([self.data.loc[idx, l] for l in self.label_names], -1))
        if len(labels) > self.length:
            labels = labels[:self.length]
        else:
            labels = torch.cat([labels, torch.zeros(self.length - len(labels), 5)])
        
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
    def __init__(self, config, pretrained=False):
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load(config.pretrained_path, map_location='cpu'))
        self.decoder = nn.Linear(256, 5)

    def forward(self, src):
        sequence_features, pairwise_features = self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        output = self.decoder(sequence_features)
        return output.squeeze(-1)

# Training and evaluation functions
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, save_path, schedule=None):
    best_loss = np.inf
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        tbar = tqdm(train_loader)
        total_loss = 0
        for idx, batch in enumerate(tbar):
            sequence = batch['sequence'].to(device)
            labels = batch['labels'].float().to(device)
            seq_length = batch['length'].to(device)
            output = model(sequence)
            
            final_output, final_labels = slice_outputs(output, labels, seq_length)
            
            loss = criterion(final_output, final_labels).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1)}")
        
        if schedule:
            schedule.step()
            
        model.eval()
        val_loss, val_preds = evaluate_model(model, val_loader, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            best_preds = val_preds
            torch.save(model.state_dict(), save_path)

def evaluate_model(model, val_loader, criterion):
    val_loss = 0
    val_preds = []
    device = next(model.parameters()).device
    tbar = tqdm(val_loader)
    for idx, batch in enumerate(tbar):
        sequence = batch['sequence'].to(device)
        labels = batch['labels'].float().to(device)
        with torch.no_grad():
            output = model(sequence)
            loss = criterion(output[:, :68], labels).mean()
        val_loss += loss.item()
        val_preds.append([labels.cpu().numpy(), output.cpu().numpy()])
    val_loss /= len(tbar)
    print(f"Validation loss: {val_loss}")
    return val_loss, val_preds

def main(args):
    set_seed()
    
    # Load configuration and initialize model
    config = load_config_from_yaml(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = finetuned_RibonanzaNet(config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Load and process data
    data, data_noisy, test107, test130 = load_data(args.train_pseudo_path, args.test_pseudo_107_path, args.test_pseudo_130_path, args.noisy_threshold)
    train_split, val_split = split_data(data)

    # Update pseudo labels for noisy data
    data_noisy = update_pseudo_labels(data_noisy)
    test107 = update_pseudo_labels(test107)
    test130 = update_pseudo_labels(test130)
    train_split = augment_real_with_pseudo(train_split)

    train_step3, highSN = prepare_training_data(train_split, data_noisy, test107, test130, args.sn_threshold)

    # Create data loaders
    train_loader3 = DataLoader(RNA_Dataset(train_step3, args.max_seq_length), batch_size=32, shuffle=True)
    highSN_loader = DataLoader(RNA_Dataset(highSN, args.max_seq_length), batch_size=32, shuffle=True)
    val_loader = DataLoader(RNA_Dataset(val_split, args.max_seq_length), batch_size=32, shuffle=False)

    # Initial training with pseudo labels
    optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_model(model, train_loader3, val_loader, epochs=args.epochs, optimizer=optimizer, criterion=MCRMAE, save_path=args.save_path)

    # Annealed training with high SN data
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    optimizer = Ranger(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(highSN_loader))
    train_model(model, highSN_loader, val_loader, epochs=args.epochs, optimizer=optimizer, criterion=MCRMAE, save_path=args.save_path, schedule=schedule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RibonanzaNet with pseudo labels")
    parser.add_argument("--config_path", type=str, default="configs/pairwise.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--model_path", type=str, default="RibonanzaNet-Deg_21.pt", help="Path to the initial model state dict")
    parser.add_argument("--train_pseudo_path", type=str, default="train_pseudo.json", help="Path to the train pseudo JSON file")
    parser.add_argument("--test_pseudo_107_path", type=str, default="test_pseudo_107.json", help="Path to the test pseudo 107 JSON file")
    parser.add_argument("--test_pseudo_130_path", type=str, default="test_pseudo_130.json", help="Path to the test pseudo 130 JSON file")
    parser.add_argument("--save_path", type=str, default="RibonanzaNet-Deg_30.pt", help="Path to save the trained model state dict")
    parser.add_argument("--sn_threshold", type=float, default=5.0, help="Signal-to-noise threshold for high SN filtering")
    parser.add_argument("--noisy_threshold", type=float, default=1.0, help="Threshold for noisy data filtering")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--max_seq_length", type=int, default=130, help="Maximum sequence length")

    args = parser.parse_args()
    main(args)
