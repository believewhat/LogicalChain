from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher
import torch
import ipdb
import os
import json
import re
import csv
import pandas as pd
import numpy as np



jsonl_data1 = []
for i in range(10178):
    jsonl_data1.append(pd.read_json(f"/home/jwang/Project/doctorrobot/LongLoRA/usmle_save_temp/{i}_reason.json", lines=True))
    jsonl_data1.append(pd.read_json(f"/home/jwang/Project/doctorrobot/LongLoRA/usmle_save_temp/{i}.json", lines=True))

data2 = pd.DataFrame(jsonl_data1)
data2['id'] = np.zeros(data2.shape[0], int)


amboss_path = "amboss_train_doc.json"

with open(amboss_path, 'r') as file:
    jsonl_data2 = json.load(file)

data3 = pd.DataFrame(jsonl_data2)[["input", "output"]]

import re



data3['id'] = np.zeros(data3.shape[0], int)


data3 = data3[['id', 'input']]
data2 = data2[['id', 'input']]

jsonl_data_amboss = []
for i in range(data3.shape[0]):
    data3['input'].loc[i] = data3['input'].loc[i] + '\nCorrect Answer: ' + data3['output'].loc[i]
    data3['id'].loc[i] = i + data.shape[0] + data2.shape[0]



data_summary = pd.concat([data[['id', 'input']], data2, data3])
data_summary['id'] = range(data_summary.shape[0])
data_summary['input'] = data_summary['input'].str.replace('\n', ' ', regex=False)


ipdb.set_trace()