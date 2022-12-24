#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:21:12 2022

@author: will
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import numpy as np
from torch.utils.data import Dataset,DataLoader


emb_dims,emb_card = [4,8,16,16,32,4],[5,12,31,24,60,7]
d_raw = 7
X_len, y_len = 512,64

# =============================================================================
# Data
# =============================================================================
class CustomDataset(Dataset):
    def __init__(self, X, date, y, X_len, y_len):
        self.X = X
        self.y = y
        self.date = date
        self.length = X.shape[0]
        self.X_len = X_len
        self.y_len = y_len
        self.seq_len = X_len + y_len

    def __len__(self):
        return (self.length//self.seq_len) - 1

    def __getitem__(self, idx):
        idx_batch = idx * self.seq_len
        return self.X[idx_batch:idx_batch+self.X_len],self.date[idx_batch:idx_batch+self.X_len],\
                self.y[idx_batch+self.X_len:idx_batch+self.X_len+self.y_len]


def get_data(batchSize):
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    date_train = np.load('date_train.npy')
    date_val = np.load('date_val.npy')
    
    data_val = DataLoader(CustomDataset(X_val,date_val,y_val, X_len, y_len),batchSize,False)
    data_train = DataLoader(CustomDataset(X_train,date_train,y_train, X_len, y_len),batchSize,True)
    return data_val,data_train




# =============================================================================
# Model
# =============================================================================


class ODEWrap(nn.Module):
    def __init__(self, odefunc):
        super(ODEWrap, self).__init__()
        self.odefunc = odefunc

    def forward(self, x, integration_time):
        out = odeint_adjoint(self.odefunc, x, integration_time)
        return out[1:]

class func_time(nn.Module):
    def __init__(self, d, layers, BN, time_fun):
        # time_fun: t -> f(t) to cat to x
        super(func_time, self).__init__()
        networks = []
        self.time_fun = time_fun
        for _ in range(layers):
            if BN:
                networks.append(nn.BatchNorm1d(d))
            networks.append(nn.Linear(d,d))
            networks.append(nn.Mish())
        self.networks = nn.Sequential(*networks[:-1])
    
    def forward(self,t,x):
        x += self.time_fun(t,x)
        return self.networks(x)

class attention(nn.MultiheadAttention):
    def forward(self,x):
        return super(attention,self).forward(x,x,x,need_weights=False)[0]

# model = func_time(8,1,True,lambda t,x: torch.ones(x.shape[-1])*t)
# model(0.5,torch.rand(10,8)).shape

class func_depth(nn.Module):
    def __init__(self, d, layers, num_heads, BN, dropout, time_fun):
        super(func_depth, self).__init__()
        networks = []
        self.time_fun = time_fun
        for _ in range(layers):
            if BN:
                networks.append(nn.LayerNorm(d))
            if dropout>0:
                networks.append(nn.Dropout(dropout))
            networks.append(attention(d, num_heads))
            networks.append(nn.Linear(d,d))
            networks.append(nn.Mish())
        self.networks = nn.Sequential(*networks[:-1])
    
    def forward(self,t,x):
        x += self.time_fun(t,x)
        return self.networks(x)

# d, d_target, layers_time, layers_depth, num_heads, BN, dropout, T = 16,7,2,2,4,True,0.1,12
# time_fun = lambda t,x: torch.ones(x.shape[-1],device=x.device)*t

class ODE_Decoder(nn.Module):
    def __init__(self, d, d_target, layers_time, layers_depth, num_heads, BN, dropout, time_fun,T):
        # T is the # of prediction time points
        super(ODE_Decoder, self).__init__()
        self.model_time = ODEWrap(func_time(d, layers_time, BN, time_fun))
        self.model_depth = ODEWrap(func_depth(d, layers_depth, num_heads, BN, dropout, time_fun))
        self.model_output = nn.Sequential(nn.Linear(d,d_target),nn.Mish())
        self.integration_time = torch.linspace(0,1,T+1).to('cuda') # +1 to account for input
        self.integration_depth = torch.linspace(0,1,2).to('cuda')
        
    def forward(self,x):
        x = self.model_time(x,self.integration_time)
        x = self.model_depth(x,self.integration_depth)
        return self.model_output(x[0]) # T,N,d_target
    
class ODE_Encoder(nn.Module):
    def __init__(self, d, layers, num_heads, BN, dropout, time_fun):
        # T is the # of prediction time points
        super(ODE_Encoder, self).__init__()
        self.model_depth = ODEWrap(func_depth(d, layers, num_heads, BN, dropout, time_fun))
        self.integration_depth = torch.linspace(0,1,2).to('cuda')
        
    def forward(self,x):
        x = self.model_depth(x,self.integration_depth)
        return x[0] # T,N,d


# TODO
class ODE_timeSeries(nn.Module):
    def __init__(self, d, d_target, layers_time, layers_depth, num_heads, BN, dropout, time_fun,T,\
                 d_enc, layers_enc, num_heads_enc, BN_enc, dropout_enc):
        # encoder_d is a tuple (87,128)
        super(ODE_timeSeries, self).__init__()
        self.encoder = ODE_Encoder(d_enc, layers_enc, num_heads_enc, BN_enc, dropout_enc, time_fun)
        self.decoder = ODE_Decoder(d, d_target, layers_time, layers_depth, num_heads, BN, dropout, time_fun,T)
        self.embed = nn.ModuleList([nn.Embedding(card,d) for d,card in zip(emb_dims,emb_card)])
        self.model_input = nn.Sequential(nn.Linear(np.sum(emb_dims)+d_raw,d_enc),nn.Mish())
    
    def forward(self,x,date):
        # x,date have shape (N,T,d)
        out = torch.cat([e(date[:,:,i]) for i,e in enumerate(self.embed)]+[x],2)
        out = self.model_input(out).transpose(0,1) 
        out = self.encoder(out) # T,N,d






















