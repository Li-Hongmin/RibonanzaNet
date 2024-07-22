# %%
#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %%
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random


from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
from sklearn.model_selection import KFold, StratifiedKFold


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


# %% [markdown]
# #训练方案分析
# 原模型是在sn>1的样本上进行finetune训练的。
# 
# 但是对于sn较差的样本没有理会，
# - 应该在这些样本上进行semi-supervise。
# - 最后在真实样本上再进行退火处理。

# %% [markdown]
# # Get data and do some data processing

# %%
data=pd.read_json("//work/gs58/d58004/datasets/openVaccine/train.json",lines=True)
test_data=pd.read_json("//work/gs58/d58004/datasets/openVaccine/test.json",lines=True)
data_noisy = data.loc[data['signal_to_noise']<=1].reset_index(drop=True)
data=data.loc[data['signal_to_noise']>1].reset_index(drop=True)
# #data=data.loc[data['length']<400].reset_index(drop=True)
data.shape, data_noisy.shape, test_data.shape

# %% [markdown]
# # Split train data into train/val
# 
# We will split the data with stratified kfold on seq length>100 to ensure a good amount of long/short sequences in train/val

# %%
data['length']=data['sequence'].apply(len)
kf = StratifiedKFold(n_splits=10,random_state=2020, shuffle=True)
for i, (train_index, val_index) in enumerate(kf.split(data,data['SN_filter'])):
    break

train_split=data.loc[train_index].reset_index(drop=True)
val_split=data.loc[val_index].reset_index(drop=True)
val_split=val_split.loc[val_split['signal_to_noise']>1].reset_index(drop=True)

plt.hist(train_split['signal_to_noise'],bins=30)
plt.hist(val_split['signal_to_noise'],bins=30)
plt.xlabel('signal_to_noise')

# %% [markdown]
# # Get pytorch dataset

# %%
# %%
train_dataset=RNA_Dataset(train_split)
val_dataset=RNA_Dataset(val_split)
noisy_dataset=RNA_Dataset(data_noisy)
test_dataset=RNA_test_Dataset(test_data)

# %%
train_dataset[0]['labels'].shape

# %%
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
noisy_loader=DataLoader(noisy_dataset,batch_size=16,shuffle=False)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)


# %% [markdown]
# # Get RibonanzaNet
# We will add a linear layer to predict RNA degradation

# %%
! pip install einops

# %%
import sys

sys.path.append("/work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final")


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



    

# %%
config=load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml")
model=finetuned_RibonanzaNet(config,pretrained=True).cuda()

# %%
#1. Initial Model Training-only confident labels:
model.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet-Deg.pt",map_location='cpu'))

# %% [markdown]
# # Pseudo label

# %%
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=False)
noisy_loader=DataLoader(noisy_dataset,batch_size=16,shuffle=False)
test_loader=DataLoader(test_data,batch_size=16,shuffle=False)

# %%
from tqdm import tqdm

pseudo_labels=[]
model.eval()
for batch in tqdm(noisy_loader):
    sequence=batch['sequence'].cuda()

    with torch.no_grad():
        pseudo_labels.extend(model(sequence).cpu().numpy())
pseudo_labels = np.array(pseudo_labels)

data_noisy_pseudo_label = data_noisy.copy()
data_noisy_pseudo_label["reactivity"] = pseudo_labels[:,:,0].tolist()
data_noisy_pseudo_label["deg_Mg_pH10"] = pseudo_labels[:,:,0].tolist()
data_noisy_pseudo_label["deg_pH10"] = pseudo_labels[:,:,0].tolist()
data_noisy_pseudo_label["deg_Mg_50C"] = pseudo_labels[:,:,0].tolist()
data_noisy_pseudo_label["deg_50C"] = pseudo_labels[:,:,0].tolist()
data_noisy_pseudo_label


# %%
from tqdm import tqdm

pseudo_labels=[]
model.eval()
for batch in tqdm(train_loader):
    sequence=batch['sequence'].cuda()

    with torch.no_grad():
        pseudo_labels.extend(model(sequence).cpu().numpy())
pseudo_labels = np.array(pseudo_labels)

train_pseudo_label = train_split.copy()
train_pseudo_label["reactivity"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_Mg_pH10"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_pH10"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_Mg_50C"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_50C"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label


# %%
from tqdm import tqdm

pseudo_labels=[]
model.eval()
for batch in tqdm(test_loader):
    sequence=batch['sequence'].cuda()

    with torch.no_grad():
        pseudo_labels.extend(model(sequence).cpu().numpy())
pseudo_labels = np.array(pseudo_labels)

train_pseudo_label = test_data.copy()
train_pseudo_label["reactivity"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_Mg_pH10"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_pH10"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_Mg_50C"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label["deg_50C"] = pseudo_labels[:,:,0].tolist()
train_pseudo_label


# %% [markdown]
# # Training loop


# %% [markdown]
# 2. Second Model Training-pseudo labels for noise labels:
# 
# Used predictions from the RibonanzaNet-Deg as pseudo-labels for the remaining noisy training data (304 sequences).
# Pre-trained RibonanzaNet for 20 epochs with a flat learning rate of 0.001.
# Followed by 10 epochs of training on true labels of high signal-to-noise sequences using a cosine learning rate schedule.
# 

# %%
train_dataset_2stage = pd.concat([train_split,data_noisy_pseudo_label]).reset_index(drop=True)

class RNA_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.label_names=['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
#         print(self.data.loc[idx,'sequence'])
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)
        
        labels=np.stack([self.data.loc[idx,l] for l in self.label_names],-1)
        labels=torch.tensor(labels)
        
        

        return {'sequence':sequence,
                'labels':labels[:68]}
    
train_dataset2=RNA_Dataset(train_dataset_2stage)
train_loader2=DataLoader(train_dataset2,batch_size=16,shuffle=True)
from tqdm import tqdm
tbar=tqdm(train_loader2)
for i in tbar:
    pass

# %%
import sys
sys.path.append('/kaggle/working/Ranger-Deep-Learning-Optimizer/ranger')
from ranger import Ranger
from tqdm import tqdm
#loss function
def MCRMAE(y_pred, y_true):
    # 计算每列的MAE
    colwise_mae = torch.mean(torch.abs(y_true - y_pred), dim=0)
    # 计算列MAE的均值
    MCRMAE = torch.mean(colwise_mae)
    return MCRMAE

epochs=20
cos_epoch=15

best_loss=np.inf
optimizer = Ranger(model.parameters(), lr=0.001)


criterion=MCRMAE
lr = 0.001
# schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs-cos_epoch)*len(train_loader))


for epoch in range(epochs):
    model.train()
    tbar=tqdm(train_loader2)
    total_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        labels=batch['labels'].cuda()

        output=model(sequence)

        loss=criterion(output[:,:68],labels)
        loss=loss.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        optimizer.zero_grad()

#         if (epoch+1)>cos_epoch:
#             schedule.step()
        #schedule.step()
        total_loss+=loss.item()

        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)}")
        #break

    tbar=tqdm(val_loader)
    model.eval()
    val_preds=[]
    val_loss=0
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        labels=batch['labels'].cuda()

        with torch.no_grad():
            output=model(sequence)
            
            loss=criterion(output[:,:68],labels)
            loss=loss.mean()
        val_loss+=loss.item()
        val_preds.append([labels.cpu().numpy(),output.cpu().numpy()])
    val_loss=val_loss/len(tbar)
    print(f"val loss: {val_loss}")
    if val_loss<best_loss:
        best_loss=val_loss
        best_preds=val_preds
        torch.save(model.state_dict(),'RibonanzaNet-Deg_2.pt')

    # 1.053595052265986 train loss after epoch 0

# %%
import sys
sys.path.append('/kaggle/working/Ranger-Deep-Learning-Optimizer/ranger')
from ranger import Ranger
from tqdm import tqdm
#loss function
def MCRMAE(y_pred, y_true):
    # 计算每列的MAE
    colwise_mae = torch.mean(torch.abs(y_true - y_pred), dim=0)
    # 计算列MAE的均值
    MCRMAE = torch.mean(colwise_mae)
    return MCRMAE

epochs=10
cos_epoch=7

best_loss=np.inf
optimizer = Ranger(model.parameters(), weight_decay=0.001, lr=0.001)


criterion=MCRMAE
lr = 0.001
schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs-cos_epoch)*len(train_loader))


for epoch in range(epochs):
    model.train()
    tbar=tqdm(train_loader)
    total_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        labels=batch['labels'].cuda()

        output=model(sequence)

        loss=criterion(output[:,:68],labels)
        loss=loss.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1)>cos_epoch:
            schedule.step()
        #schedule.step()
        total_loss+=loss.item()

        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)}")
        #break

    tbar=tqdm(val_loader)
    model.eval()
    val_preds=[]
    val_loss=0
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        labels=batch['labels'].cuda()

        with torch.no_grad():
            output=model(sequence)
            
            loss=criterion(output[:,:68],labels)
            loss=loss.mean()
        val_loss+=loss.item()
        val_preds.append([labels.cpu().numpy(),output.cpu().numpy()])
    val_loss=val_loss/len(tbar)
    print(f"val loss: {val_loss}")
    if val_loss<best_loss:
        best_loss=val_loss
        best_preds=val_preds
        torch.save(model.state_dict(),'RibonanzaNet-Deg.pt')

    # 1.053595052265986 train loss after epoch 0

# %% [markdown]
# # stage 3 
# 
# 这时，要把所有位置的伪标签全部计算出来，包括测试和training的数据中的伪标签。
# 先训练，再真实退火。
# 
# Final Model Training-pseduo labels for all noisy labels and test sequences + annealed on high confident labels:
# 
# Created a new pseudo-label dataset of 1,907,619 sequences using predictions from the top 3 Kaggle models for noisy labels and all test sequences.
# Repeated training on this larger pseudo-label dataset and then annealed on true high signal-to-noise data.
# This phase took 140 hours on 10xL40S GPUs.
# 

# %%



