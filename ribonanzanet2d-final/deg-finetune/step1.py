# %%
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random


# %%
#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% [markdown]
# #训练方案分析
# 原模型是在sn>1的样本上进行finetune训练的。
# 
# 但是对于sn较差的样本没有理会，
# - 应该在这些样本上进行semi-supervise。
# - 最后在真实样本上再进行退火处理。

# %% [markdown]
# # Get data and do some data processing

# %% [markdown]
# # Get pytorch dataset

# %%
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval


from Network import *
import yaml



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",map_location='cpu'))
        self.decoder=nn.Linear(256,5)

    def forward(self,src):
        
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        output=self.decoder(sequence_features)

        return output.squeeze(-1)
    

class RNA_test_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in self.data.loc[idx,'sequence']]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        return {'sequence':sequence}

class RNA_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.label_names=['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)
        
        labels=np.stack([self.data.loc[idx,l] for l in self.label_names],-1)
        labels=torch.tensor(labels)
        
        return {'sequence':sequence,
                'labels':labels}

# %%
config=load_config_from_yaml("/Users/lihongmin/Research/ideas/RibonanzaNet/ribonanzanet2d-final/configs/pairwise.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device= torch.device("mps")
model=finetuned_RibonanzaNet(config,pretrained=False).to(device)
#1. Initial Model Training-only confident labels:
model.load_state_dict(torch.load("/Users/lihongmin/Research/ideas/RibonanzaNet/ribonanzanet-weights/RibonanzaNet-Deg.pt",map_location=device))

# %%
data=pd.read_json("/Users/lihongmin/Research/24 mRNAdegredation/RNAdegformer/src/OpenVaccine/data/train.json",lines=True).reset_index(drop=True)
# data_noisy = data.loc[data['signal_to_noise']<=1].reset_index(drop=True)
# data=data.loc[data['signal_to_noise']>1].reset_index(drop=True)
test_data=pd.read_json("/Users/lihongmin/Research/24 mRNAdegredation/RNAdegformer/src/OpenVaccine/data/test.json",lines=True).reset_index(drop=True)
# #data=data.loc[data['length']<400].reset_index(drop=True)
data.shape, test_data.shape
test_data_107 = test_data.loc[test_data['seq_length']==107].reset_index(drop=True)
test_data_130 = test_data.loc[test_data['seq_length']==130].reset_index(drop=True)

# %%
train_dataset=RNA_test_Dataset(data)
test_dataset=RNA_test_Dataset(test_data)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
test_loader_107=DataLoader(RNA_test_Dataset(test_data_107),batch_size=32,shuffle=False)
test_loader_130=DataLoader(RNA_test_Dataset(test_data_130),batch_size=32,shuffle=False)

# %%
from tqdm import tqdm

pseudo_labels=[]
model.eval()
for batch in tqdm(train_loader):
    sequence=batch['sequence'].to(device)
    with torch.no_grad():
        pseudo_labels.extend(model(sequence).cpu().numpy())
pseudo_labels = np.array(pseudo_labels)

data_pseudo_label = data.copy()
data_pseudo_label["pseudo_reactivity"] = pseudo_labels[:,:,0].tolist()
data_pseudo_label["pseudo_deg_Mg_pH10"] = pseudo_labels[:,:,1].tolist()
data_pseudo_label["pseudo_deg_pH10"] = pseudo_labels[:,:,2].tolist()
data_pseudo_label["pseudo_deg_Mg_50C"] = pseudo_labels[:,:,3].tolist()
data_pseudo_label["pseudo_deg_50C"] = pseudo_labels[:,:,4].tolist()
data_pseudo_label.to_json("train_pseudo.json",orient='records',lines=True)


# %%
from tqdm import tqdm

pseudo_labels=[]
model.eval()
for batch in tqdm(test_loader_107):
    sequence=batch['sequence'].to(device)
    with torch.no_grad():
        pseudo_labels.extend(model(sequence).cpu().numpy())
pseudo_labels = np.array(pseudo_labels)

test_data_107_pseudo_label = test_data_107.copy()
test_data_107_pseudo_label["pseudo_reactivity"] = pseudo_labels[:,:,0].tolist()
test_data_107_pseudo_label["pseudo_deg_Mg_pH10"] = pseudo_labels[:,:,1].tolist()
test_data_107_pseudo_label["pseudo_deg_pH10"] = pseudo_labels[:,:,2].tolist()
test_data_107_pseudo_label["pseudo_deg_Mg_50C"] = pseudo_labels[:,:,3].tolist()
test_data_107_pseudo_label["pseudo_deg_50C"] = pseudo_labels[:,:,4].tolist()
test_data_107_pseudo_label.to_json("test_pseudo_107.json",orient='records',lines=True)

# test_data_130
pseudo_labels=[]
model.eval()
for batch in tqdm(test_loader_130):
    sequence=batch['sequence'].to(device)
    with torch.no_grad():
        pseudo_labels.extend(model(sequence).cpu().numpy())
pseudo_labels = np.array(pseudo_labels)

test_data_130_pseudo_label = test_data_130.copy()
test_data_130_pseudo_label["pseudo_reactivity"] = pseudo_labels[:,:,0].tolist()
test_data_130_pseudo_label["pseudo_deg_Mg_pH10"] = pseudo_labels[:,:,1].tolist()
test_data_130_pseudo_label["pseudo_deg_pH10"] = pseudo_labels[:,:,2].tolist()
test_data_130_pseudo_label["pseudo_deg_Mg_50C"] = pseudo_labels[:,:,3].tolist()
test_data_130_pseudo_label["pseudo_deg_50C"] = pseudo_labels[:,:,4].tolist()
test_data_130_pseudo_label.to_json("test_pseudo_130.json",orient='records',lines=True)



# %%



