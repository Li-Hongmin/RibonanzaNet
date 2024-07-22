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
    def __init__(self,data, length=68):
        self.data=data
        self.length=length
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.label_names=['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)
        if len(sequence)<self.length:
            sequence=torch.cat([sequence,torch.ones(self.length-len(sequence)).long()*4])
        labels=np.stack([self.data.loc[idx,l] for l in self.label_names],-1)
        labels=torch.tensor(labels)
        if len(labels)>self.length:
            labels=labels[:self.length]
        else:
            labels=torch.cat([labels,torch.zeros(self.length-len(labels),5)])
        return {'sequence':sequence,
                'labels':labels}

# %%
config=load_config_from_yaml("/Users/lihongmin/Research/ideas/RibonanzaNet/ribonanzanet2d-final/configs/pairwise.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device= torch.device("mps")
model=finetuned_RibonanzaNet(config,pretrained=False).to(device)
#1. Initial Model Training-only confident labels:
model.load_state_dict(torch.load("/Users/lihongmin/Research/ideas/RibonanzaNet/ribonanzanet-weights/RibonanzaNet-Deg21.pt",map_location=device))

# %%
data=pd.read_json("train_pseudo.json",lines=True).reset_index(drop=True)
data_noisy = data.loc[data['signal_to_noise']<=1].reset_index(drop=True)
data=data.loc[data['signal_to_noise']>1].reset_index(drop=True)
test107 = pd.read_json("test_pseudo_107.json",lines=True).reset_index(drop=True)
test130 = pd.read_json("test_pseudo_130.json",lines=True).reset_index(drop=True)

# %%
from sklearn.model_selection import KFold, StratifiedKFold
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
# # copy pseudo label for noisy data

# %%
data_noisy.columns

# %%
data_noisy[['reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']] = data_noisy[['pseudo_reactivity',
       'pseudo_deg_Mg_pH10', 'pseudo_deg_pH10', 'pseudo_deg_Mg_50C',
       'pseudo_deg_50C']]

test107[['reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']] = test107[['pseudo_reactivity',
       'pseudo_deg_Mg_pH10', 'pseudo_deg_pH10', 'pseudo_deg_Mg_50C',
       'pseudo_deg_50C']]
test130[['reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']] = test130[['pseudo_reactivity',
       'pseudo_deg_Mg_pH10', 'pseudo_deg_pH10', 'pseudo_deg_Mg_50C',
       'pseudo_deg_50C']]

# 将train_split的前68个值保留原样，后面的值用pseudo的值替代
# train_step3=pd.concat([train_split,data_noisy,test107,test130],axis=0).reset_index(drop=True)
# train_step3
train_split['reactivity'] = train_split.apply(lambda x: x['reactivity'] + x['pseudo_reactivity'][68:], axis=1)
train_split['deg_Mg_pH10'] = train_split.apply(lambda x: x['deg_Mg_pH10'] + x['pseudo_deg_Mg_pH10'][68:], axis=1)
train_split['deg_pH10'] = train_split.apply(lambda x: x['deg_pH10'] + x['pseudo_deg_pH10'][68:], axis=1)
train_split['deg_Mg_50C'] = train_split.apply(lambda x: x['deg_Mg_50C'] + x['pseudo_deg_Mg_50C'][68:], axis=1)
train_split['deg_50C'] = train_split.apply(lambda x: x['deg_50C'] + x['pseudo_deg_50C'][68:], axis=1)
train_step3=pd.concat([train_split,data_noisy,test107,test130],axis=0).reset_index(drop=True)


# %%
highSN=train_step3.loc[train_step3['signal_to_noise']>5].reset_index(drop=True)
highSN

# %%
class RNA_Dataset(Dataset):
    def __init__(self,data, length=68):
        self.data=data
        self.length=length
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.label_names=['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)
        seq_len=len(sequence)
        if len(sequence)<self.length:
            sequence=torch.cat([sequence,torch.ones(self.length-len(sequence)).long()*4])
        labels=np.stack([self.data.loc[idx,l] for l in self.label_names],-1)
        labels=torch.tensor(labels)
        if len(labels)>self.length:
            labels=labels[:self.length]
        else:
            labels=torch.cat([labels,torch.zeros(self.length-len(labels),5)])
        return {'sequence':sequence,
                "length": seq_len,
                'labels':labels}
train_loader3=DataLoader(RNA_Dataset(train_step3,130),batch_size=32,shuffle=True)
highSN_loader=DataLoader(RNA_Dataset(highSN),batch_size=32,shuffle=True)
val_loader = DataLoader(RNA_Dataset(val_split),batch_size=32,shuffle=False)

# %%
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
    tbar=tqdm(train_loader3)
    total_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].to(device)
        labels=batch['labels'].float().to(device)
        seq_length=batch['length'].to(device)
        output=model(sequence)
        
        sliced_outputs = []
        slice_labels = []       

        for i in range(output.size(0)):  # Loop through each batch
            # Slice each batch up to its corresponding sequence length
            sliced_output = output[i, :seq_length[i], :]
            sliced_label = labels[i, :seq_length[i], :]
            # Append the sliced output to the list
            sliced_outputs.append(sliced_output)
            slice_labels.append(sliced_label)

        # Concatenate all sliced outputs along the first dimension
        final_output = torch.cat(sliced_outputs, dim=0)
        final_labels = torch.cat(slice_labels, dim=0)
        
        loss=criterion(final_output,final_labels)
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
        sequence=batch['sequence'].to(device)
        labels=batch['labels'].float().to(device)

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
        torch.save(model.state_dict(),'RibonanzaNet-Deg_30.pt')

    # 1.053595052265986 train loss after epoch 0

# %%
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
schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs-cos_epoch)*len(highSN_loader))


for epoch in range(epochs):
    model.train()
    tbar=tqdm(highSN_loader)
    total_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].to(device)
        labels=batch['labels'].float().to(device)

        output=model(sequence)

        loss=criterion(output[:,:68],labels)
        loss=loss.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1)>cos_epoch:
            schedule.step()
        total_loss+=loss.item()

        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)}")
        #break

    tbar=tqdm(val_loader)
    model.eval()
    val_preds=[]
    val_loss=0
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].to(device)
        labels=batch['labels'].float().to(device)

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
        torch.save(model.state_dict(),'RibonanzaNet-Deg_31.pt')

    # 1.053595052265986 train loss after epoch 0


