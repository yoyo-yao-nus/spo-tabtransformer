# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:37:11 2025

@author: Yoyooooo
"""

import torch, numpy as np, os, pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer



class TabularDataset(Dataset):

    DIRECTORY: str = '/content/drive/MyDrive/Colab Notebooks/spo_ttf'

    def __init__(self, df, zip=5, train = True, convert_target=False):
        if convert_target:
          import ast
          df['Target']=df['Target'].apply(ast.literal_eval)
          df.set_index('index',inplace=True)
        self.df = df
        # self.categorical_cols = categorical_cols
        self.categorical_cols = df.columns[9:]
        self.numerical_cols = [col for col in df.columns if col not in self.categorical_cols and col != 'Target']
        self.target_col = 'Target'
        # contraints block
        self.constrs = np.load(os.path.join(self.DIRECTORY,'Data/constrs.npy'))
        self.d = np.load(os.path.join(self.DIRECTORY,'Data/d.npy'))
        if train:
          self.zip5 = np.load(os.path.join(self.DIRECTORY,'Data/train_zip5.npy'))
          self.zip4 = np.load(os.path.join(self.DIRECTORY,'Data/train_zip4.npy'))
          self.zip3 = np.load(os.path.join(self.DIRECTORY,'Data/train_zip3.npy'))
        else:
          self.zip5 = np.load(os.path.join(self.DIRECTORY,'Data/val_zip5.npy'))
          self.zip4 = np.load(os.path.join(self.DIRECTORY,'Data/val_zip4.npy'))
          self.zip3 = np.load(os.path.join(self.DIRECTORY,'Data/val_zip3.npy'))
    
        self.lmbdas = np.load(os.path.join(self.DIRECTORY,'Data/lmbdas.npy'))
        self.gammas = np.load(os.path.join(self.DIRECTORY,'Data/gammas.npy'))
        self.seg_count = np.load(os.path.join(self.DIRECTORY,'Data/seg_count.npy'))
        self.unique_zip5 = np.unique(self.zip5)
        self.x_true = np.load(os.path.join(self.DIRECTORY,'Data/x_true.npy'))

        if zip == 5:
          self.seg = self.zip5
          self.unique_seg = self.unique_zip5


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_num = torch.tensor(row[self.numerical_cols].values.astype(float), dtype=torch.float32)
        x_cat = torch.tensor(row[self.categorical_cols].values.astype(int))
        y = torch.tensor(row[self.target_col], dtype=torch.float32)
        seg = self.seg[idx]
        constrs = self.constrs[idx]
        d = self.d[idx]
        constrs_dict = {
            'seg':seg,
            'constrs':constrs,
            'd':d
        }
        return x_num, x_cat, y, constrs_dict


class RatioDataloader():
    def __init__(self, tabular_dataset, sample_ratio=0.005):
        self.dataset = tabular_dataset
        self.sample_ratio = sample_ratio
        self.segments = tabular_dataset.seg
        self.unique_segments = np.unique(self.segments)
        self.seg_count = tabular_dataset.seg_count

        self.constrs = tabular_dataset.constrs
        self.d = tabular_dataset.d
        self.categorical_cols = tabular_dataset.categorical_cols
        self.numerical_cols = tabular_dataset.numerical_cols
        self.target_col = tabular_dataset.target_col

        self.x_num = torch.tensor(self.dataset.df[self.numerical_cols].values.astype(float), dtype=torch.float32)
        self.x_cat = torch.tensor(self.dataset.df[self.categorical_cols].values.astype(int))
        self.y = torch.tensor(np.array(self.dataset.df[self.target_col].tolist()), dtype=torch.float32)
        self.constrs_dict = {
            'seg':self.segments,
            'constrs':self.constrs,
            'd':self.d
        }

        self.segment_indices = {
            seg: np.where(self.segments == seg)[0]
            for seg in self.unique_segments
        }
        self.segment_sample_sizes = {
            seg: max(1, int(len(indices) * sample_ratio))
            for seg, indices in self.segment_indices.items()
        }

        # initialize sampling history
        self.reset_sampling_history()

        # correct the number of batches
        self.max_batches_per_segment = min(
            len(indices) // self.segment_sample_sizes[seg]
            for seg, indices in self.segment_indices.items()
        )
        
    def __len__(self):
        return self.max_batches_per_segment

    def reset_sampling_history(self):
        self.sampled_indices = {
            segment: set() for segment in self.unique_segments
        }

    def get_segment_batch(self):
        batch = []
        batch_indices = np.array([])

        for segment in self.unique_segments:
            segment_indices = self.segment_indices[segment]
            sample_size = self.segment_sample_sizes[segment]

            available_indices = list(
                set(range(len(segment_indices))) - self.sampled_indices[segment]
            )

            if len(available_indices) < sample_size:
                self.sampled_indices[segment] = set()
                available_indices = list(range(len(segment_indices)))
            # local indices for the list 'segment_indices'
            chosen_local_indices = np.random.choice(
                available_indices,
                size=sample_size,
                replace=False
            )

            # update the sampling history
            self.sampled_indices[segment].update(chosen_local_indices)

            # global indices for the real dataset grouped by segment
            chosen_global_indices = segment_indices[chosen_local_indices]
            batch_indices = np.append(batch_indices,chosen_global_indices)
            x_num = self.x_num[chosen_global_indices]
            x_cat = self.x_cat[chosen_global_indices]
            y = self.y[chosen_global_indices]
            constrs_dict = {
                'seg':self.segments[chosen_global_indices],
                'constrs':self.constrs[chosen_global_indices],
                'd':self.d[chosen_global_indices]
            }
            batch.append((x_num, x_cat, y, constrs_dict))
        batch_x_num = torch.vstack([seg_data[0] for seg_data in batch])
        batch_x_cat = torch.vstack([seg_data[1] for seg_data in batch])
        batch_y = torch.vstack([seg_data[2] for seg_data in batch])
        batch_seg = np.concatenate([seg_data[3]['seg'] for seg_data in batch])
        batch_constrs = np.concatenate([seg_data[3]['constrs'] for seg_data in batch])
        batch_d = np.concatenate([seg_data[3]['d'] for seg_data in batch])
        constrs_dict = {
            'seg':batch_seg,
            'constrs':batch_constrs,
            'd':batch_d
        }
        batch = (batch_x_num, batch_x_cat, batch_y, constrs_dict)
        return batch

    def __iter__(self):
        self.current_batch = 0
        self.reset_sampling_history()
        self.total_batches = self.max_batches_per_segment
        return self

    def __next__(self):
        if self.current_batch >= self.total_batches:
            raise StopIteration

        self.current_batch += 1
        return self.get_segment_batch()
   
categories = np.load(os.path.join(TabularDataset.DIRECTORY,'Data/categories.npy'))    
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device) for v in batch)
    else:
        return batch
    

from tqdm import tqdm

def train_spo(spo_model, train_dataloader, epochs):
    step = 0
    for epoch in tqdm(range(epochs)):
      spo_model.model.train()
      epoch_loss = []
      for x_num, x_cat, y, constrs_dict in tqdm(train_dataloader):
        step += 1
        (x_num, x_cat, y) = move_batch_to_device((x_num, x_cat, y), device=device)
        spo_loss = spo_model.update(x_cat, x_num, y, constrs_dict)
        epoch_loss.append(spo_loss.item())
        if step % 10 == 0:
          print(f'Step: {step}, SPO Loss: {spo_loss}')
        if step % 200 == 0:
          file_name = os.path.join(TabularDataset.DIRECTORY,'spo_ttf_mpax.pth')
          torch.save(spo_model,file_name)

        # print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

      spo_loss = np.mean(epoch_loss)
      print(f'Epoch: {epoch}, SPO Loss: {spo_loss}')

def ranking_acc(dataloader,model,spo_model):
  acc_ttf = 0
  acc_spo = 0
  minmax_acc_ttf = 0
  minmax_acc_spo = 0
  minmax_sum = 0
  sum = 0
  for x_num, x_cat, y, constrs_dict in tqdm(dataloader):
    (x_num, x_cat, y) = move_batch_to_device((x_num, x_cat, y), device=device)
    with torch.no_grad():
      results1 = model(x_cat, x_num)
      results2 = spo_model(x_cat, x_num, constrs_dict)
    for i, yi in enumerate(y.detach().cpu().numpy()):
      y_sorted = sorted(enumerate(yi),key=lambda x:x[1])
      y_ttf_sorted = sorted(enumerate(results1[i].detach().cpu().numpy()),key=lambda x:x[1])
      y_spo_sorted = sorted(enumerate(results2[i].detach().cpu().numpy()),key=lambda x:x[1])
      acc_ttf += np.sum(np.array(y_sorted)[:,0]==np.array(y_ttf_sorted)[:,0])
      acc_spo += np.sum(np.array(y_sorted)[:,0]==np.array(y_spo_sorted)[:,0])
      sum += len(y_sorted)

      minmax_acc_ttf += (y_sorted[0][0] == y_ttf_sorted[0][0]) + (y_sorted[-1][0] == y_ttf_sorted[-1][0])
      minmax_acc_spo += y_sorted[0][0] == y_spo_sorted[0][0] + (y_sorted[-1][0] == y_spo_sorted[-1][0])
      minmax_sum += 2

      # print(f"True ranking: {[item[0] for item in y_sorted]}")
      # print(f"TTF ranking: {[item[0] for item in y_ttf_sorted]}")
      # print(f"SPO ranking: {[item[0] for item in y_spo_sorted]}")
      # print()
  acc_ttf /= sum
  acc_spo /= sum
  print(f"TTF accuracy: {acc_ttf}")
  print(f"SPO accuracy: {acc_spo}")
  minmax_acc_ttf /= minmax_sum
  minmax_acc_spo /= minmax_sum
  print(f"TTF minmax accuracy: {minmax_acc_ttf}")
  print(f"SPO minmax accuracy: {minmax_acc_spo}")
  return acc_ttf, acc_spo, minmax_acc_ttf, minmax_acc_spo

import torchmetrics
def evaluate(dataloader,model,spo_model,flag='train'): # flag='test'
  with torch.no_grad():
      metric1 = torchmetrics.MeanSquaredError().to(device)
      metric2 = torchmetrics.MeanSquaredError().to(device)
      for x_num, x_cat, y, constrs_dict in tqdm(dataloader):
          x_num, x_cat, y = move_batch_to_device((x_num, x_cat, y), device=device)
          preds = model(x_cat, x_num)
          spo_preds = spo_model(x_cat, x_num, constrs_dict)
          metric1(preds, y)
          metric2(spo_preds, y)
  mse1 = metric1.compute()
  mse2 = metric2.compute()
  print(f'Prediction MSE on {flag} data: {mse1}\nSPO MSE on {flag} data: {mse2}')
  return mse1, mse2

