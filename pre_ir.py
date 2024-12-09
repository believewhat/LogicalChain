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
import sys


if __name__ ==  '__main__':
    data_file = 'usmle_train_ir.json'
    collection = './open_guideline/open_medguideline_noqa.tsv'
    num_doc = 20
    with open(data_file, "r") as file:
        data = json.load(file)
    
    start_index = int(sys.argv[1])
    #end_index = int(sys.argv[2])
    query_file = sys.argv[2]
    save_file = sys.argv[3]
    
    
    data = pd.DataFrame(data).loc[start_index:]
    data.reset_index(inplace=True, drop=True)

    data['id'] = np.zeros(data.shape[0], int)
    for i in range(data.shape[0]):
        data['id'].loc[i] = i+1
        data['input'].loc[i] = data['input'].loc[i].replace('\n', ' ')
    
    data.dropna(inplace=True)

    def count_words(text):
        return len(text.split())

        # 应用这个函数到 'input' 列，创建一个布尔索引，筛选出单词数不超过500的行
    data = data[data['input'].apply(count_words) <= 500]

    data[['id', 'input']].to_csv(query_file, index=False, header=False, sep='\t')
    
    
    nbits = 2
    checkpoint_path = "/home/jwang/Project/doctorrobot/LongLoRA/colbert/colbertv2.0"
    experiment_root = "./experiments/"
    experiment_name = "open_medguideline_noqa"
    index_name = f'{experiment_name}.{nbits}bits'
    
    with Run().context(RunConfig(nranks=4, root=experiment_root, experiment=experiment_name)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=512, query_maxlen=128, nbits=nbits)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    

    
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        config = ColBERTConfig(
            root="./",
            doc_maxlen=512, 
            query_maxlen=128, 
            nbits=nbits
        )
        searcher = Searcher(index=index_name, checkpoint=checkpoint_path, config=config)
        queries = Queries(query_file)
        import ipdb
        ipdb.set_trace()
        
        #rank1 = searcher.search(queries.data[1], k=num_doc)



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
                
                if doc_dict[x[1]] in data['input'].loc[i] or data['input'].loc[i] in doc_dict[x[1]]:
                    continue
                
                doc_list.append(doc_dict[x[1]])
            print(len(doc_list))
            output.append({'input': data['input'].loc[i], 'output': data['output'].loc[i], 'doc': doc_list})
        with open(save_file, 'w') as file:
            json.dump(output, file, indent=4)

