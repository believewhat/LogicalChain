# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
os.environ["WANDB_DISABLED"] = "true"
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast

from llama_attn_replace_sft import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model, PeftModel
from torch.distributed import barrier
import ipdb
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from colbert.data import Collection, Queries, Ranking
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")
    peft_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

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


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

experiment_name = "llama_ir"

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    index_name: str
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s for s in sources]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:1] = IGNORE_INDEX
    
    
    
    with Run().context(RunConfig(experiment=experiment_name)):
        searcher = Searcher(index=index_name)
    
    Q = searcher.encode(sources, full_length_search=False)

    return dict(input_ids=input_ids, labels=labels, input_ids_query=Q, text=sources)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        list_data_dict2 = jload("./data/uptodate.json")
        list_data_dict3 = jload("./data/textbook.json")
        list_data_dict4 = jload("./data/mayoclinical.json")
        list_data_dict5 = jload("./data/mimic4-full_dev.json")
        logging.warning("Formatting inputs...")
        '''
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        '''
        #sources = [example["text"] for example in list_data_dict]
        sources = [example["text"] for example in list_data_dict+list_data_dict2+list_data_dict3+list_data_dict4+list_data_dict5]
        logging.warning("Tokenizing inputs... This may take some time...")

        data_dict = preprocess(sources, tokenizer, index_name)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.input_ids_query = data_dict["input_ids_query"]
        self.text = data_dict["text"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], input_ids_query=self.input_ids_query[i], text=self.text[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, input_ids_query, text = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "input_ids_query", "text"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            input_ids_query=input_ids_query,
            text=text,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs,):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control



class MyCustomModel(PreTrainedModel):
    def __init__(self, colbert, lm_model, tokenizer, doc, doc_token):
        super().__init__(lm_model.config)
        self.colbert = colbert
        self.lm_model = lm_model
        self.tok = tokenizer
        self.doc = doc
        self.doc_token = doc_token
    def cal_loss_logit(self, 
        input_ids: torch.LongTensor = None,
        input_ids_lens: torch.LongTensor = None,
        text: str = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            label=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.lm_model.config.pretraining_tp > 1:
            lm_head_slices = self.lm_model.lm_head.weight.split(self.lm_model.vocab_size // self.lm_model.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.lm_model.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_model.lm_head(hidden_states)
        logits = logits.float()
        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        return loss, logits, outputs
        
    def forward(self, 
        input_ids: torch.LongTensor = None,
        input_ids_query: torch.LongTensor = None,
        text: str = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        Q = self.colbert.query(*self.query_token) # torch.Size([16, 127, 128])
        bs = Q.shape[0]
        D, D_mask = self.doc(*self.doc_token, keep_dims='return_mask') # torch.Size([32, 512, 128]), torch.Size([32, 512, 1])

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(bs, dim=0).contiguous() # torch.Size([32, 127, 128])
        D = D.repeat((bs,1,1))
        D_mask = D_mask.repeat((bs,1,1))

        scores = self.colbert.score(Q_duplicated, D, D_mask)

        

        doc_pid, _, scores = self.searcher.dense_search(input_ids_query, 10, filter_fn=None, pids=None)

        doc_ref = ""
        for i in range(len(doc_pid)):
            doc_ref += f"Material{i}:\n{self.doc[doc_pid[i]]}\n\n"

        text_input = f"{doc_ref}\n\n{text}"
        input_ids_summ, labels_summ, input_ids_lens_summ, labels_lens_summ = _tokenize_fn([text_input], self.tok)


        input_ids_summ = torch.nn.utils.rnn.pad_sequence(
            input_ids_summ, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_summ = torch.nn.utils.rnn.pad_sequence(labels_summ, batch_first=True, padding_value=IGNORE_INDEX)


        loss2, logits2, outputs2 = self.cal_loss_logit(
                input_ids=input_ids_summ,
                attention_mask=input_ids_summ.ne(self.tokenizer.pad_token_id),
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                label=labels_summ,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )


        

        """
        loss1, logits1, outputs1 = self.cal_loss_logit(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            label=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        indices = torch.multinomial(scores, num_samples=10, replacement=True)

        sampled_doc = [doc_pid[index] for index in indices]

        loss_ir = 0

        for i in range(len(sampled_doc)):
            doc_ref = f"Material:\n{self.doc[sampled_doc[i]]}\n\n"

            text_input = f"{doc_ref}\n\n{text}"
            input_ids_summ, labels_summ, input_ids_lens_summ, labels_lens_summ = _tokenize_fn([text_input], self.tok)


            input_ids_summ = torch.nn.utils.rnn.pad_sequence(
                input_ids_summ, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels_summ = torch.nn.utils.rnn.pad_sequence(labels_summ, batch_first=True, padding_value=IGNORE_INDEX)
        
            loss2, logits2, outputs2 = self.cal_loss_logit(
                input_ids=input_ids_summ,
                attention_mask=input_ids_summ.ne(self.tokenizer.pad_token_id),
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                label=labels_summ,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            loss_ir += logits1 / logits2




        if not return_dict:
            output = (logits2,) + outputs2[1:]
            return (loss2,) + output if loss2 is not None else output

        return CausalLMOutputWithPast(
            loss=loss2,
            logits=logits2,
            past_key_values=outputs2.past_key_values,
            hidden_states=outputs2.hidden_states,
            attentions=outputs2.attentions,
        )
        """


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print('peft_dir')
    print(model_args.peft_dir)
    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn) 
    else:
        replace_llama_attn(training_args.use_flash_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

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

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]
        if not model_args.peft_dir:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=targets,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        else:
            model = PeftModel.from_pretrained(model, model_args.peft_dir, is_trainable=True)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module, callbacks=[SavePeftModelCallback])
    if not model_args.peft_dir:
        trainer.train(resume_from_checkpoint=None)
    else:
        trainer.train(resume_from_checkpoint=model_args.peft_dir)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    dataset = experiment_name
    datasplit = 'test_trained' #TODO: change
    collection = './downloads/mimic_dis/col_train.tsv'
    
    doc_maxlen = 512   # truncate passages at 512 tokens
    query_maxlen = 128   # truncate passages at 512 tokens
    nbits = 2   # encode each dimension with 2 bits
    index_name = f'{dataset}.{datasplit}.{nbits}bits'
    checkpoint_path='./experiments/mimic_dis__v5/none/2022-09/22/08.10.39/checkpoints/colbert'

    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, query_maxlen=query_maxlen, nbits=nbits)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    
    with Run().context(RunConfig(experiment=experiment_name)):
        searcher = Searcher(index=index_name)
    train()
