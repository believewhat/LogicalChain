import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
from llama_attn_replace import replace_llama_attn
import json
import io
import ipdb
from typing import Optional, Dict, Sequence
from peft import (
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
    PeftConfig,
)
from llama_ir import LlamaIRForCausalLM
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_DOC_START_TOKEN = "<doc_start>"
DEFAULT_DOC_END_TOKEN = "<doc_end>"
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
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--colbert_model', type=str, default="/home/zhichaoyang/mimic3/genrank/downloads/colbertv2.0")
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

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

def main(args):
    if args.flash_attn:
        replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    if not args.peft_path:
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and args.context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        config.colbert_checkpoint = None
        colbert_config = ColBERTConfig(
            doc_maxlen=128, 
            query_maxlen=512,
            bsize=4,
            lr=3e-6, 
            accumsteps=1,
            use_ib_negatives=False
        )
        colbert_config.bsize = colbert_config.bsize // colbert_config.nranks

        model.colbert = ColBERT(name=args.colbert_checkpoint, colbert_config=colbert_config)
        

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            padding_side="right",
            use_fast=False,
        )



        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
    else:
        config = PeftConfig.from_pretrained(args.peft_path)
        # config.base_model_name_or_path = ''
        config_base = transformers.AutoConfig.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                  config=config_base,
                                  trust_remote_code=True,
                                  torch_dtype=torch.float16,
                                  device_map={"":0}
        )
        model = PeftModel.from_pretrained(model, args.peft_path)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    model.eval()

    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=True)


    list_data_dict = jload('amboss_test2.json')
    ipdb.set_trace()
    sources = ["Here are some materials to help you answer the question" + '<doc_start><doc_end>'.join(example['doc'][:7])[11:-9] + Prompt + example['input'] +'\nCorrect Answer:' for example in list_data_dict]
    results = []

    


    for x in sources:
        output = respond(prompt=x)
        results.append(output)
    outputs = [example['output_option'][-2] for example in list_data_dict]
    acc = 0
    for i in range(len(results)):
        if results[i][0] == outputs[i]:
            acc += 1
        else:
            print(results[i][0], outputs[i])
    ipdb.set_trace()
if __name__ == "__main__":
    args = parse_config()
    main(args)