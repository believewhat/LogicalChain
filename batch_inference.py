# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys, math
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Seq2SeqTrainer,
)
from datasets import load_dataset
import evaluate

from peft import (
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
    PeftConfig,
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


Prompt = """
You should ask these questions and here are some examples:
A female newborn delivered at 37 weeks\u2019 gestation develops respiratory distress immediately after birth. She was delivered vaginally to a 31-year-old woman, gravida 1, para 1. Pregnancy was complicated by gestational diabetes mellitus treated with insulin during the third trimester. The newborn's pulse is 136/min, respirations are 57/min, and blood pressure is 60/35 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 91% when the newborn is crying and a saturation of 85% at rest. Examination shows grunting breath sounds and perioral blue discoloration that improves when the patient cries. Lungs are clear to auscultation. Cardiac examination shows a split S2 during inspiration but no murmurs are heard. Femoral pulses are palpable bilaterally. A suction catheter cannot be passed through the nares. In addition to establishing an oral airway, which of the following is the most appropriate treatment for this patient's condition?\nA\nArterial switch procedure\nB\nEndoscopic resection of the posterior nasal septum\nC\nReconnection of the upper esophageal pouch with the lower esophagus\nD\nAnastomosis between subclavian and pulmonary artery\nE\nEndotracheal administration of artificial surfactant
Correct Answer: B
A 2-year-old girl is brought to the physician by her parents because of clumsiness and difficulty walking. She began to walk at 12 months and continues to have difficulty standing still without support. She also appears to have difficulty grabbing objects in front of her. Over the past year, she has had 5 episodes of sinusitis requiring antibiotic treatment and was hospitalized twice for bacterial pneumonia. Physical examination shows an unstable, narrow-based gait and several hyperpigmented skin patches. Serum studies show decreased levels of IgA and IgG and an increased level of alpha-fetoprotein. Over the next 5 years, which of the following complications is this patient most likely to develop?\nA\nChronic eczema\nB\nConjunctival telangiectasias\nC\nPes cavus\nD\nCardiac rhabdomyoma\nE\nCeliac disease\nF\nChronic lymphocytic leukemia
Correct Answer: B
"""

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    context_size: Optional[int] = field(
        default=4096,
        metadata={"help": "Context length of the model for longlora: https://github.com/dvlab-research/LongLoRA/"}
    )
    do_rope_scaling: Optional[bool] = field(
        default=False,
        metadata={"help": "Do RoPE scalling or not for longlora: https://github.com/dvlab-research/LongLoRA/"}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    mmlu_med_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    peft_path: str = field(default=None, metadata={"help": 'Path to peft model if any'})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'}) 

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True) 

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0) 
    repetition_penalty: Optional[float] = field(default=1.0) 
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0) 

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def get_accelerate_model(args, checkpoint_dir):
    # n_gpus = torch.cuda.device_count()
    # max_memory = f'{args.max_memory_MB}MB'
    # max_memory = {i: max_memory for i in range(n_gpus)}
    # device_map = 'auto'
    device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}

    if args.full_finetune: assert args.bits in [16, 32]

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    config_base = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    if args.do_rope_scaling:
        context_size = args.context_size
        orig_ctx_len = getattr(config_base, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config_base.rope_scaling = {"type": "linear", "factor": scaling_factor}

    if not args.peft_path is None:
        print(f'loading peft model {args.peft_path}...')
        config = PeftConfig.from_pretrained(args.peft_path)
        # config.base_model_name_or_path = ''
        print(f'loading base model {config.base_model_name_or_path}...')
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                  config=config_base,
                                  trust_remote_code=True,
                                  torch_dtype=torch.float16,
                                  device_map={"":0}
        )
        model = PeftModel.from_pretrained(model, args.peft_path)
        # model = model.merge_and_unload()
    else:
        print(f'loading base model {args.model_name_or_path}...')
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.model_name_or_path,
        #     config=config_base,
        #     cache_dir=args.cache_dir,
        #     load_in_4bit=True,
        #     device_map={"":0}
        # )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config_base,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16
        )


    return model

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")

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

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [Prompt+example['input']+'\nCorrect Answer: ' for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = [] 
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'], 
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict



PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')
    print(checkpoint_dir)
    model = get_accelerate_model(args, checkpoint_dir)
    training_args.skip_loading_checkpoint_weights=True

    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print('loaded model')
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # if any(key in args.model_name_or_path for key in ['llama', '7B', '13B', '30B', '65B']):
    #     # LLaMA tokenizer does not have special tokens set.
    #     # Add them to prevent them from being parsed into different tokens.
    #     # Note that these are present in the vocabulary. 
    #     # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
    #     tokenizer.add_special_tokens(
    #         {
    #             "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
    #             "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
    #             "unk_token": tokenizer.convert_ids_to_tokens(model.config.pad_token_id), 
    #         }
    #     )

    # data_module = make_data_module(tokenizer=tokenizer, args=args)
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer, 
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
        
    trainer = Seq2SeqTrainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )

    mmlu_dataset = load_dataset("json", data_files={'test': 'amboss_test_correct.json'})
    mmlu_dataset = mmlu_dataset['test']
    accuracy = evaluate.load("accuracy")

    data_loader = trainer.get_eval_dataloader(mmlu_dataset)
    source_max_len = trainer.data_collator.source_max_len
    trainer.data_collator.source_max_len = args.mmlu_source_max_len
    trainer.model.eval()
    preds, refs = [], []
    abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
            tokenizer("E", add_special_tokens=False).input_ids[0],
            tokenizer("F", add_special_tokens=False).input_ids[0],
            tokenizer("G", add_special_tokens=False).input_ids[0],
            tokenizer("H", add_special_tokens=False).input_ids[0],
            tokenizer("I", add_special_tokens=False).input_ids[0],
        ]
    for batch in tqdm(data_loader, total=len(data_loader)):
        (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
        # There are two tokens, the output, and eos token.
        try:
            labels_new = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0] #TODO
            refs += [abcd_idx.index(label) for label in labels_new.tolist()]
        except:
            continue
        for i, logit in enumerate(logits):
            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
            logit_abcd = logit[label_non_zero_id-1][abcd_idx]
            preds.append(torch.argmax(logit_abcd).item())
        
    # Extract results by subject.
    results = {}
    subjects = {'refs':[], 'preds':[]}
    for p,r in zip(preds, refs):
        subjects['preds'].append(p)
        subjects['refs'].append(r)
    # subject_med_scores = []
    subject_score = accuracy.compute(
        references=subjects['refs'],
        predictions=subjects['preds']
    )['accuracy']
    results[f'mmlu_{args.mmlu_split}_accuracy'] = subject_score
    print(results)
    # # Training
    # if args.do_train:
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
    #     metrics = train_result.metrics
    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()
    #     all_metrics.update(metrics)
    # # Evaluation
    # if args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate(metric_key_prefix="eval")
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)
    #     all_metrics.update(metrics)
    # # Prediction
    # if args.do_predict:
    #     logger.info("*** Predict ***")
    #     prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
    #     prediction_metrics = prediction_output.metrics
    #     predictions = prediction_output.predictions
    #     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    #     predictions = tokenizer.batch_decode(
    #         predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #     )
    #     with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
    #         for i, example in enumerate(data_module['predict_dataset']):
    #             example['prediction_with_input'] = predictions[i].strip()
    #             example['prediction'] = predictions[i].replace(example['input'], '').strip()
    #             fout.write(json.dumps(example) + '\n')
    #     print(prediction_metrics)
    #     trainer.log_metrics("predict", prediction_metrics)
    #     trainer.save_metrics("predict", prediction_metrics)
    #     all_metrics.update(prediction_metrics)

    # if (args.do_train or args.do_eval or args.do_predict):
    #     with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
    #         fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
