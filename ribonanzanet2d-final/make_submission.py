# %%
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

# %% [markdown]
# # Define Dataset

# %%
from torch.utils.data import Dataset, DataLoader

class RNA2D_Dataset(Dataset):
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

# %%
test_data=pd.read_json("/work/gs58/d58004/datasets/openVaccine/test.json",lines=True)
test_dataset=RNA2D_Dataset(test_data)
test_dataset[0]

# %%
len(test_data)

# %% [markdown]
# # Define Model

# %%


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
    def __init__(self, config):
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.decoder = nn.Linear(config.ninp,5)
        
    def forward(self,src):
        
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        output=self.decoder(sequence_features) #predict

        return output#.squeeze(-1)

# %%
config=load_config_from_yaml("configs/pairwise.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device= torch.device("mps")
model=finetuned_RibonanzaNet(config).to(device)
#1. Initial Model Training-only confident labels:
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--para', type=str, default="RibonanzaNet-Deg_31.pt")

args = parser.parse_args()
model.load_state_dict(torch.load(args.para,map_location=device))

# multi-gpu 
if torch.cuda.device_count() > 1:
    
    import torch.nn as nn
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
# %% [markdown]
# # Make predictions

# %%
from tqdm import tqdm

test_preds=[]
model.eval()
for i in tqdm(range(len(test_dataset))):
    example=test_dataset[i]
    sequence=example['sequence'].to(device).unsqueeze(0)

    with torch.no_grad():
        test_preds.append(model(sequence).cpu().numpy())

# %%
# let's take a look at the predictions
import matplotlib.pyplot as plt
labels=['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
for i in range(5):
    
    plt.plot(test_preds[0][0,:,i],label=labels[i])
    
plt.legend()


# %%
preds=[]
ids=[]
for i in range(len(test_data)):
    preds.append(test_preds[i][0,:])
    id=test_data.loc[i,'id']
    ids.extend([f"{id}_{pos}" for pos in range(len(test_preds[i][0,:]))])
    #break
preds=np.concatenate(preds)
preds.shape

# %%
sub=pd.DataFrame()

sub['id_seqpos']=ids

for i,l in enumerate(['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']):
    sub[l]=preds[:,i]

name = args.para.split('/')[-1]
dir = 'submissions'
sub.to_csv(f'{dir}/submission_{name}.csv',index=False)

# %%


