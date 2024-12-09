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

ulmse_path = "medmcqa_reason.json"

with open(ulmse_path, 'r') as file:
    jsonl_data1 = json.load(file)

data = pd.read_csv('uptodate.tsv', sep='\t', header=None)
data2 = pd.DataFrame(jsonl_data1)
data2['id'] = np.zeros(data2.shape[0], int)
data2['input'] = data2.apply(lambda row: (row['input'] + '\nCorrect Answer: ' + row['output']).replace('\n', ' '), axis=1)

# 更新 'id' 列
data2['id'] = range(data.shape[0], data.shape[0] + data2.shape[0])

data2 = data2[['id', 'input']]
data2.columns = [0, 1]
data_summary = pd.concat([data, data2])
data_summary[0] = range(data_summary.shape[0])


data_summary.to_csv('medmcqa_collection.tsv', sep='\t', index=False, header=False)
ipdb.set_trace()

if __name__ ==  '__main__':
    data_file = 'medmcqa_reason.json'
    query_file = "query_medmmcqa.tsv"
    save_file = 'medmcqa_train_doc.json'
    collection = 'medmcqa_collection.tsv'
    num_doc = 30
    with open(data_file, "r") as file:
        data = json.load(file)
    data = pd.DataFrame(data)
    data['id'] = range(1, data.shape[0]+1)
    data['input'] = data.apply(lambda row: row['input'].replace('\n', ' '), axis=1)
    data[['id', 'input']].to_csv(query_file, index=False, header=False, sep='\t')

    nbits = 2
    checkpoint_path = "ir"
    experiment_root = "./experiments/"
    experiment_name = "guideline_qa_our"
    index_name = f'{experiment_name}.{nbits}bits'
    
    
    with Run().context(RunConfig(nranks=10, root=experiment_root, experiment=experiment_name)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=512, query_maxlen=128, nbits=nbits)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    

    
    with Run().context(RunConfig(nranks=10, experiment=experiment_name)):
        config = ColBERTConfig(
            root="./",
        )
        searcher = Searcher(index=index_name, checkpoint=checkpoint_path, config=config)
        queries = Queries(query_file)
        
        ranking = searcher.search_all(queries, k=num_doc)
        ranks_result = ranking.tolist()
        output = []
        with open(collection, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')
            doc = [row for row in reader]
        doc_dict = {}
        for x in doc:
            doc_dict[int(x[0])] = x[1]
        for i in range(data.shape[0]):
            doc_list = []
            for x in ranks_result[i*num_doc:(i+1)*num_doc]:
                if doc_dict[x[1]] in data['input'].loc[i]:
                    continue
                doc_list.append(doc_dict[x[1]])
            print(len(doc_list))
            output.append({'input': data['input'].loc[i], 'output': data['output'].loc[i], 'doc': doc_list})
        with open(save_file, 'w') as file:
            json.dump(output, file, indent=4)