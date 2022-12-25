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
import time
import copy
from torch.nn.utils import clip_grad_value_


emb_dims,emb_card = [4,8,16,16,32,4],[5,12,31,24,60,7]
d_raw = 7
X_len, y_len = 512,64
device = 'cuda' #'cpu'
max_len = 50 # # of batches for each epoch of training

# =============================================================================
# Data
# =============================================================================
class CustomDataset(Dataset):
    def __init__(self, X, date, y, X_len, y_len, max_len=None):
        self.X = X
        self.y = y
        self.date = date
        self.length = X.shape[0]
        self.X_len = X_len
        self.y_len = y_len
        self.seq_len = X_len + y_len
        self.max_len = max_len # dont run though all examples in training in one epoch

    def __len__(self):
        return (self.length//self.seq_len) - 1 if self.max_len is None else self.max_len

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
    data_train = DataLoader(CustomDataset(X_train,date_train,y_train, X_len, y_len,max_len),batchSize,True)
    return data_val,data_train # N,T,d




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
        self.integration_time = torch.linspace(0,1,T+1).to(device) # +1 to account for input
        self.integration_depth = torch.linspace(0,1,2).to(device)
        
    def forward(self,x):
        x = self.model_time(x,self.integration_time)
        x = self.model_depth(x,self.integration_depth)
        return self.model_output(x[0]) # T,N,d_target
    
class ODE_Encoder(nn.Module):
    def __init__(self, d, layers, num_heads, BN, dropout, time_fun):
        super(ODE_Encoder, self).__init__()
        self.model_depth = ODEWrap(func_depth(d, layers, num_heads, BN, dropout, time_fun))
        self.integration_depth = torch.linspace(0,1,2).to(device)
        
    def forward(self,x):
        x = self.model_depth(x,self.integration_depth)
        return x[0] # T,N,d

class Agg(nn.Module):
    # T,N,d -> N,d'
    def __init__(self, T, d_in, d_out, dropout, d_mid=2):
        super(Agg, self).__init__()
        self.lin1 = nn.Linear(T,d_mid)
        self.d_save = d_mid*d_in
        self.lin2 = nn.Linear(self.d_save,d_out)
        self.activation = nn.Mish()
        self.BN = nn.LayerNorm(d_in)
        if dropout>0:
           self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.drop(self.BN(x))
        x = self.activation(self.lin1(x.permute((1,2,0))))
        x = self.activation(self.lin2(x.view((-1,self.d_save))))
        return x
    
class ODE_timeSeries(nn.Module):
    def __init__(self, d, d_target, layers_time, layers_depth, num_heads, BN, dropout, time_fun,T,\
                 d_enc, layers_enc, num_heads_enc, BN_enc, dropout_enc,
                 d_mid):
        # encoder_d is a tuple (87,128)
        super(ODE_timeSeries, self).__init__()
        self.encoder = ODE_Encoder(d_enc, layers_enc, num_heads_enc, BN_enc, dropout_enc, time_fun)
        self.decoder = ODE_Decoder(d, d_target, layers_time, layers_depth, num_heads, BN, dropout, time_fun,T)
        self.embed = nn.ModuleList([nn.Embedding(card,d) for d,card in zip(emb_dims,emb_card)])
        self.model_input = nn.Sequential(nn.Linear(np.sum(emb_dims)+d_raw,d_enc),nn.Mish())
        self.agg = Agg(X_len,d_enc,d,dropout,d_mid)
        self.loss = nn.L1Loss()
        
    def forward(self,data):
        # x,date have shape (N,T,d)
        if len(data) == 3:
            x,date,y = data
        else:
            x,date = data
        out = torch.cat([e(date[:,:,i]) for i,e in enumerate(self.embed)]+[x],2)
        out = self.model_input(out).transpose(0,1) 
        out = self.encoder(out) # T,N,d
        out = self.agg(out) # N,d'
        out = self.decoder(out) # T,N,d_target
        out = out.permute((1,0,2))
        return self.loss(out,y) if len(data)==3 else out



def train(opt,model,epochs,train_dl,val_dl,paras,clip,verbose=True,save=False):
    since = time.time()
    lossBest = 1e6
    opt.zero_grad()
    for epoch in range(epochs):
        # training #
        model.train()
        train_loss = 0
        val_loss = 0
        
        for i,data in enumerate(train_dl):
            data = [i.to('cuda') for i in data]
            loss = model(data)
            loss.backward()
            clip_grad_value_(paras,clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
            
        # evaluating #
        model.eval()
        with torch.no_grad():
            for j,data in enumerate(val_dl):
                data = [i.to('cuda') for i in data]
                loss = model(data)
                val_loss += loss.item()
        
        # save model
        if val_loss<lossBest:
            lossBest = val_loss
            if save: bestWeight = copy.deepcopy(model.state_dict())   
        if verbose:
            print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f} \n'.format(epoch,train_loss/i,val_loss/j))

    
    if save: model.load_state_dict(bestWeight)
    time_elapsed = time.time() - since
    if verbose: print('Training completed in {}s'.format(time_elapsed))
    return model,lossBest

















