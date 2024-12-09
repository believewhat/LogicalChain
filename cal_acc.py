import json
import ipdb

import json
import ipdb
import glob
import os
import ipdb
import os
import json
import csv
import pandas as pd
import argparse
import base64
import openai
import time
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
from colbert.modeling.reranker.tokenizer import RerankerTokenizer
from typing import List, Optional, Tuple, Union

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from colbert.modeling.colbert import colbert_score
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher
import torch
import concurrent.futures
"""
openai.api_key = "sk-GtGtq9SRSDRcj1yXEf57F6B02c85413381D33cCf69DaEeB1"
openai.api_base = "https://api.xiaoai.plus/v1"
"""
openai.api_key = "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"
def apply_chatgpt(messages, temperature=0.1, max_tokens=10, presence_penalty=0, frequency_penalty=0, method="gpt-3.5-turbo"):
  cnt = 0
  while cnt < 4:
    try:
        completion = openai.ChatCompletion.create(
            model=method,
            messages=messages,
            temperature=temperature
        )
        content = completion.choices[0].message.content
        return content
    except:
        cnt += 1
        continue
  ipdb.set_trace()
  return content

def process_source(x):
    messages = [
        {"role": "system", "content": "You are a doctor now you should answer the following question based on the references I gave you."},
        {"role": "user", "content": x},
        {"role": "user", "content": "Provide your answer only return the option: A or B or C or D..."}
    ]
    result = apply_chatgpt(messages)
    return result
files = ['mmlu_open_test.json']
for file_name in files:
    if file_name == 'zero_shot_mmlu_test.json':
        med_tasks = ["clinical_knowledge", "medical_genetics", "anatomy", "professional_medicine", "college_biology", "college_medicine", "virology", "high_school_biology", "nutrition"]
        data = []
    
    
        with open('zero_shot_mmlu_test.json', 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    if json_obj['subject'] not in med_tasks:
                        continue
                    data.append(json_obj)
                except:
                    continue
    else:
        with open(file_name, "r") as file:
            data = json.load(file)
    #doc = [example["doc"] for example in data]
    
    colbert_config = ColBERTConfig(
        doc_maxlen=128, 
        query_maxlen=512,
        bsize=4,
        lr=3e-6, 
        accumsteps=1,
        use_ib_negatives=False
    )
    colbert_config.checkpoint = '/data/experiment_data/junda/chatdoctor/llama-13b-32k-usmle-open-ir/checkpoint-1779/ir'
    #colbert_config.checkpoint = '/home/jwang/Project/doctorrobot/LongLoRA/colbert/colbertv2.0'
    colbert_config.bsize = colbert_config.bsize // colbert_config.nranks

    doc_tokenizer = DocTokenizer(colbert_config)
    query_tokenizer = QueryTokenizer(colbert_config)

    colbert = ColBERT(name=colbert_config.checkpoint, colbert_config=colbert_config).cuda()
    num_doc = 5
    
    nbits = 2
    checkpoint_path = '/home/jwang/Project/doctorrobot/LongLoRA/colbert/colbertv2.0'
    experiment_root = "./experiments/"
    experiment_name = "openguideline_medmcqa"
    index_name = f'{experiment_name}.{nbits}bits'
    collection = './open_guideline/open_guideline_medmcqa.tsv'
    """
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        config = ColBERTConfig(
            root="./",
        )
        searcher = Searcher(index=index_name, checkpoint=checkpoint_path, config=config)

    with open(collection, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t')
        doc = [row for row in reader]
    doc_dict = {}
    for x in doc:
        doc_dict[int(x[0])] = x[1]


    
    q_doc = []
    for x in data:
        x['input'] = x['input'].replace('Answer:', '')
    for example in data:
        ranking = searcher.search(example['input'], k=num_doc)
        doc_list = []
        for x in ranking[0]:
            doc_list.append(doc_dict[x])
        q_doc.append(doc_list)
    
    doc = [example[:5] for example in q_doc]
    """
    sources = []
    doc = [example["doc"] for example in data]
    doc_tokens = [doc_tokenizer.tensorize(x) for x in doc]

    query_tokens = [query_tokenizer.tensorize([example["input"]]) for example in data]
    
    ipdb.set_trace()
    
    for i, example in enumerate(data):
        doc_token = doc_tokens[i]
        query_token = query_tokens[i]
        Q = colbert.query(*query_token)
        D, D_mask = colbert.doc(*doc_token, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        scores = colbert_score(Q, D, D_mask, config=colbert_config)

        selected_indices = torch.argsort(scores)[-5:]
        doc_select = [doc[i][j] for j in selected_indices]
        
        sources.append('Here are some references to help you answer:\n' + '\nReference:\n'.join(doc_select) + '\nNow you should ask my question:\nQuestion:\n' + data[i]['input'] + '\nPlease only give the correct option. Correct Answer:')
        #sources.append('\nQuestion:\n' + data[i]['input'] + '\nPlease only give the correct option. Correct Answer:')
    
    sources = [x['input'] for x in data]
    results = []
    """
    for i, x in enumerate(sources):
        messages=[
            {"role": "system", f"content": "You are a doctor now you should answer the following question based on the references I gave you."},
            {"role": "user", "content": x},
            {"role": "user", "content": "Provide your answer only return the option: A or B or C or D or E or F..."}
        ]
        result = apply_chatgpt(messages)
        data[i]['gpt3_answer'] = result
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        # Using a dictionary to map futures to source index
        future_to_index = {executor.submit(process_source, x): i for i, x in enumerate(sources)}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            data[index]['gpt3_answer'] = future.result()
    
    file_name = file_name.split('.')[0]
    with open(f"{file_name}_gpt3_raw.json", "w") as file:
        json.dump(data, file)
acc = 0
for x in data:
    if x['output'][0] == x['gpt3_answer'][0]:
        acc += 1
    else:
        if x['output'][0] == x['gpt3_answer'][-1]:
            acc += 1
        print(x['gpt3_answer'])
ipdb.set_trace()

"""
with open("usmle_doc_open_test_gpt3_raw.json", "r") as file:
    data = json.load(file)

acc = 0
for x in data:
    if x['output'][0] == x['gpt3_answer'][0]:
        acc += 1
    else:
        if x['output'][0] == x['gpt3_answer'][-1]:
            acc += 1
            continue
        print(x['output'][0], x['gpt3_answer'])
print(acc / len(data))
"""

"""
with open("amboss_test_gpt4.json", "r") as file:
    data = json.load(file)
"""
