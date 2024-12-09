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

if __name__ ==  '__main__':
    data = pd.read_csv('data_test.csv')
    nbits = 2
    checkpoint_path = "/data/experiment_data/junda/chatdoctor/llama-7b-32k-ir-2/"
    experiment_root = "./experiments/"
    experiment_name = "usmle_test"
    index_name = f'{experiment_name}.{nbits}bits'
    collection = 'uptodate.tsv'

    with Run().context(RunConfig(nranks=6, root=experiment_root, experiment=experiment_name)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=512, query_maxlen=128, nbits=nbits)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    
    with Run().context(RunConfig(nranks=4, experiment=experiment_name)):
        config = ColBERTConfig(
            root="./",
        )
        searcher = Searcher(index=index_name, checkpoint=checkpoint_path, config=config)
        queries = Queries("query_umlse_test.tsv")

        doc_num = 10
        ranking = searcher.search_all(queries, k=doc_num)
        ranking.save("ranking.tsv")
        ranks_result = ranking.tolist()
        output = []
        with open("uptodate.tsv", 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')
            doc = [row for row in reader]
        doc_dict = {}
        for x in doc:
            doc_dict[int(x[0])] = x[1]
        for i in range(data.shape[0]):
            doc_list = []
            for x in ranks_result[i*doc_num:(i+1)*doc_num]:
                doc_list.append(doc_dict[x[1]])
            output.append({'input': data['question'].loc[i], 'output': data['answer_idx'].loc[i], 'doc': doc_list})
        with open('usmle_test_ir.json', 'w') as file:
            json.dump(output, file, indent=4)