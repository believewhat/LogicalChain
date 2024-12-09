


import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
import json
import io
import ipdb
from typing import Optional, Dict, Sequence
from peft import (
    PeftModel,
)

from typing import List, Optional, Tuple, Union
import torch.multiprocessing as mp  # 用于多进程
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_DOC_START_TOKEN = "<doc_start>"
DEFAULT_DOC_END_TOKEN = "<doc_end>"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# TODO: Wherever this is called, pass `config=`


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
Prompt = """
You should ask these questions and here are some examples:
A female newborn delivered at 37 weeks\u2019 gestation develops respiratory distress immediately after birth. She was delivered vaginally to a 31-year-old woman, gravida 1, para 1. Pregnancy was complicated by gestational diabetes mellitus treated with insulin during the third trimester. The newborn's pulse is 136/min, respirations are 57/min, and blood pressure is 60/35 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 91% when the newborn is crying and a saturation of 85% at rest. Examination shows grunting breath sounds and perioral blue discoloration that improves when the patient cries. Lungs are clear to auscultation. Cardiac examination shows a split S2 during inspiration but no murmurs are heard. Femoral pulses are palpable bilaterally. A suction catheter cannot be passed through the nares. In addition to establishing an oral airway, which of the following is the most appropriate treatment for this patient's condition?\nA\nArterial switch procedure\nB\nEndoscopic resection of the posterior nasal septum\nC\nReconnection of the upper esophageal pouch with the lower esophagus\nD\nAnastomosis between subclavian and pulmonary artery\nE\nEndotracheal administration of artificial surfactant
Correct Answer: B
A 2-year-old girl is brought to the physician by her parents because of clumsiness and difficulty walking. She began to walk at 12 months and continues to have difficulty standing still without support. She also appears to have difficulty grabbing objects in front of her. Over the past year, she has had 5 episodes of sinusitis requiring antibiotic treatment and was hospitalized twice for bacterial pneumonia. Physical examination shows an unstable, narrow-based gait and several hyperpigmented skin patches. Serum studies show decreased levels of IgA and IgG and an increased level of alpha-fetoprotein. Over the next 5 years, which of the following complications is this patient most likely to develop?\nA\nChronic eczema\nB\nConjunctival telangiectasias\nC\nPes cavus\nD\nCardiac rhabdomyoma\nE\nCeliac disease\nF\nChronic lymphocytic leukemia
Correct Answer: B
"""

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--material', type=str, default="")
    parser.add_argument('--question', type=str, default="")
    parser.add_argument('--base_model', type=str, default="/data/experiment_data/junda/Llama-2-7b-hf")
    parser.add_argument('--peft_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    args = parser.parse_args()
    return args

def read_txt_file(material_txt):
    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

def build_generator(model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True):
    def response(prompt, device):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)
        out = out.split(prompt.lstrip("<s>"))[1].strip()
        return out

    return response

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

# 并行推理函数，每个进程对应一个GPU
def inference_worker(rank, args, data_splits):
    device_id = rank  # 直接使用 rank 作为 device_id
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    config_base = transformers.AutoConfig.from_pretrained(
        args.base_model, cache_dir=args.cache_dir
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, padding_side="right", use_fast=True
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, config=config_base, trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path).to(device)
    
    model.eval()

    respond = build_generator(
        model, tokenizer, temperature=args.temperature, top_p=args.top_p,
        max_gen_len=args.max_gen_len, use_cache=True
    )

    sources = []
    directory = "/data/experiment_data/junda/datasets--MedRAG--wikipedia/snapshots/d76b9ad82135e352235d17e75921c49b68fd07b2/chunk/"
    for example in data_splits[rank]:  # 每个进程只处理属于自己的数据
        #doc_select = example["doc"][:3]
        random_file = random.choice([f for f in os.listdir(directory) if f.endswith(".jsonl")])
        file_path = os.path.join(directory, random_file)

        # 读取文件并随机选择三行
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # 随机选择三行
        selected_lines = random.sample(lines, 3)
        doc_select = []
        for line in selected_lines:
            record = json.loads(line)
            doc_select.append(record["content"])
        sources.append('Reference:\n' + '\nReference:\n'.join(doc_select) + '\nQuestion:\n' + example['input'] + '. Correct Answer:\nBioReference:')
    
    results = []
    for x in sources:
        output = respond(prompt=x, device=device)
        results.append(output)

    # 保存每个进程的推理结果到单独的JSON文件
    output_filename = f"inference_results_gpu_{rank}.json"
    save_results_to_json(results, output_filename)

    print(f"Process {rank} finished on device {device_id} with {len(results)} results.")
import random
def main(args):
    # 加载数据
    file_name = 'usmle_test_doc_our.json'
    list_data_dict = jload(file_name)
    list_data_dict = random.sample(list_data_dict, 100)
    
    # 确定使用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for parallel inference.")
    
    # 将数据按 GPU 数量划分
    chunk_size = len(list_data_dict) // num_gpus
    data_splits = [list_data_dict[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]
    
    # 使用 multiprocessing 启动多个进程
    mp.spawn(inference_worker, args=(args, data_splits), nprocs=num_gpus, join=True)
    result_splits = []
    num_gpu = 4
    for i in range(num_gpu):
        with open(f'inference_results_gpu_{i}.json', 'r') as file:
            result_splits.append(json.load(file))
    ipdb.set_trace()
    import re
    outputs = []
    
    ipdb.set_trace()
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    # 计算每对摘要之间的分数
    scores = []
    for i in range(num_gpu):
        for system, reference in zip(result_splits[i], data_splits[i]):
            score = scorer.score('\nReference:\n'.join(reference['doc'][:3]), system)
            scores.append(score)
    rouge1 = []
    rouge2 = []
    rougeL = []
    rougeLsum = []
    for x in scores:
        rouge1.append(x['rouge1'].fmeasure)
        rouge2.append(x['rouge2'].fmeasure)
        rougeL.append(x['rougeL'].fmeasure)
        rougeLsum.append(x['rougeLsum'].fmeasure)
    import numpy as np
    print(np.mean(rouge1[25:]))
    print(np.mean(rouge2))
    print(np.mean(rougeL))
    print(np.mean(rougeLsum))
    
    import spacy
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    #nlp = spacy.load("en_core_web_sm")

    def extract_entities(text):
        doc = nlp(text)
        return [ent.text for ent in doc.ents]  # 返回实体文本列表

    def compare_entities(system, reference):
        system_entities = extract_entities(system)
        reference_entities = extract_entities(reference)
    
        if not system_entities or not reference_entities:
            return False

        # 计算实体重叠情况
        common_entities = set(system_entities) & set(reference_entities)
        total_entities = set(system_entities) | set(reference_entities)
    
        # 如果实体重叠部分在95%以上，则认为两者一致
        if len(common_entities) / len(total_entities) >= 0.95:
            return True
        else:
            return False

    exact_acc = 0
    temp = []

    for i in range(num_gpu):
        for system, reference in zip(result_splits[i], data_splits[i]):
            if len(system) < 600:
                continue
            temp.append(reference)
            if '\n'.join(reference["doc"])[10:500] in system or system[10:500] in '\n'.join(reference["doc"]):
                exact_acc += 1


    for i in range(num_gpu):
        for system, reference in zip(result_splits[i], data_splits[i]):
            if len(system) < 600:
                continue
    
            references = reference.split('Reference:')
        
            flag = 0
            system = system.split('Reference:')
            for x in references:
                if len(x) < 600:
                    continue
                for y in system:
                    temp_x = x.replace('\n', '')
                    temp_y = y.replace('\n', '')
                    # 使用实体比较函数
                    if compare_entities(temp_y, temp_x):
                        exact_acc += 1
                        flag = 1
                        break
                if flag:
                    break
    
            if flag:
                temp.append((references, system))
    
        print(exact_acc / len(temp))
if __name__ == "__main__":
    args = parse_config()
    main(args)

"""
def deal():
    exact_acc = 0
    temp = []
    for i in range(4):
        for system, reference in zip(result_splits[i][5:], data_splits[i][5:]):
            if system[50:400] in '\nReference:\n'.join(reference['doc'][:3]) or '\nReference:\n'.join(reference['doc'][:3])[50:400] in system:
                exact_acc += 1
            temp.append(system)
    print(exact_acc / len(temp))
deal()

def deal():
    exact_acc = 0
    temp = []
    for i in range(4):
        for system, reference in zip(result_splits[i], data_splits[i]):
            if system == reference['output']:
                exact_acc += 1
            temp.append(system)
    print(exact_acc / len(temp))


for i in range(len(data_splits[0])):
    docs = list_data_dict[i]['doc']
    y = result_splits[0][i]
    for x in docs:
        # 检查连续公共子字符串
        def longest_common_substring(str1, str2):
            m, n = len(str1), len(str2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            longest, end_pos = 0, 0
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        if dp[i][j] > longest:
                            longest = dp[i][j]
                            end_pos = i
            return str1[end_pos - longest:end_pos], longest

        # 查找 x 和 y 的最长公共连续子字符串
        common_substring, length = longest_common_substring(x, y)
        
        # 如果公共子字符串长度占 x 的长度达到 95%，计数 +1 并退出循环
        if length >= 0.95 * len(x):
            count += 1
            print(count)
            break  # 退出当前 docs 循环

"""