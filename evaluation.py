# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:38:03 2025

@author: Yoyooooo
"""
from utils import *
import os, torch, pandas as pd
model_path = os.path.join(TabularDataset.DIRECTORY, 'models/ttf.pth')
model = torch.load(model_path,weights_only=False)
spo_model_path = os.path.join(TabularDataset.DIRECTORY, 'models/spo_ttf_mpax.pth')
spo_model = torch.load(spo_model_path,weights_only=False)

# Evaluate training data
train_df = pd.read_csv(os.path.join(TabularDataset.DIRECTORY,'train/data.csv'))
tabular_dataset = TabularDataset(train_df, convert_target=True)
train_dataloader = RatioDataloader(tabular_dataset, sample_ratio=0.0005)

acc_ttf, acc_spo, minmax_acc_ttf, minmax_acc_spo = ranking_acc(train_dataloader,model,spo_model)
mse1, mse2 = evaluate(train_dataloader,model,spo_model)

# Evaluate test data
test_df = pd.read_csv(os.path.join(TabularDataset.DIRECTORY,'validation/data.csv'))
test_dataset = TabularDataset(test_df, train=False, convert_target=True)
test_dataloader = RatioDataloader(test_dataset, sample_ratio=0.005)

acc_ttf, acc_spo, minmax_acc_ttf, minmax_acc_spo = ranking_acc(test_dataloader,model,spo_model)
mse1, mse2 = evaluate(test_dataloader,model,spo_model,flag='test')