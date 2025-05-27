# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:55:41 2025

@author: Yoyooooo
"""
from utils import *
from spo_ttf_mpax_model import SPOTTF

train_df = pd.read_csv(os.path.join(TabularDataset.DIRECTORY,'train/data.csv'))
tabular_dataset = TabularDataset(train_df, convert_target=True)
train_dataloader = RatioDataloader(tabular_dataset, sample_ratio=0.0005)

num_continuous = 8
dim_out = 6
epochs = 5
spo_model = SPOTTF(categories, num_continuous, dim_out)
train_spo(spo_model, train_dataloader, epochs=epochs)


# CUDA memory control
"""
import gc
gc.collect()                
torch.cuda.empty_cache()     
torch.cuda.ipc_collect()    

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) and obj.is_cuda:
#             print(type(obj), obj.size())
#     except:
#         pass
