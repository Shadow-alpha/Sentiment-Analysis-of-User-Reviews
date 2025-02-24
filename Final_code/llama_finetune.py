# run: srun -p mllm-align --quotatype=reserved --gres=gpu:8 --cpus-per-task=16 --time=300 accelerate launch --config_file ../configs/accelerate_configs/deepspeed_zero1.yaml train_on_cot.py
# dev: srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 accelerate launch --config_file ../configs/accelerate_configs/single_gpu.yaml train_on_cot.py

# reference: 
# https://github.com/tloen/alpaca-lora/blob/main/finetune.py
# https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

from dataclasses import dataclass
import os
import json
import torch
import pandas as pd
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from peft import get_peft_model, LoraConfig

from utils import set_seeds, print_local_main


@dataclass
class ModelArguments:
    model_path:     str = "/mnt/hwfile/llm-safety/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer_path: str = ""
    load_in_4bit:   bool = False
    use_flash_attention_2: bool = True

@dataclass
class DataArguments:
    dataset_name: str = None
    num_proc:     int = 8
    max_length:   int = 1024

@dataclass
class PeftArguments:
    use_peft:       bool  = False
    target_modules: str   = "all-linear"
    r:              int   = 16
    lora_alpha:     int   = 16
    lora_dropout:   float = 0.05
    bias:           str   = "none"
    task_type:      str   = "CAUSAL_LM"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    train_on_prompt: bool = False
    output_dir: str = "models/meta-llama/Meta-Llama-3.1-8B-Instruct-sst-finetuned"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.05
    bf16: bool = True
    run_name: str = "models/meta-llama/Meta-Llama-3.1-8B-Instruct-sst-finetuned"
    report_to: str = "wandb"
    num_train_epochs: int = 1
    logging_steps: float = 10
    save_steps: float = 0.25
    system_prompt: str = "You are a intelligent helper for sentiment classification."
    # eval_steps: float = 0.25
    # eval_delay: float = 0.25
    evaluation_strategy: str = "no"
    save_total_limit: int = 1
    # save_only_model: bool = True
    load_best_model_at_end: bool = False


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    set_seeds(training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True) if model_args.load_in_4bit and transformers.utils.is_bitsandbytes_available() else None,
        **({"device_map": {"": Accelerator().local_process_index}} if not transformers.modeling_utils.is_deepspeed_zero3_enabled else {}),
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 and transformers.utils.is_flash_attn_2_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path if not model_args.tokenizer_path else model_args.tokenizer_path, padding_side="right")
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

    # peft
    if peft_args.use_peft:
        peft_config = LoraConfig(
            r=peft_args.r,
            target_modules=peft_args.target_modules,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias=peft_args.bias,
            task_type=peft_args.task_type,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    def load_and_combine_datasets(dataset_paths):
        csv_data = pd.read_csv(dataset_paths, sep='\t',header=None)
        csv_data.columns = ['text', 'sentiment_index', 'author']
        train_dataset =  Dataset.from_pandas(csv_data)
        return train_dataset

    def process_dataset(row, label_pad_token_id=-100):
        messages = row
        # calculate prompt length
        # breakpoint()
        SENTIMENT_CLASSES = ['admiration',
        'amusement',
        'anger',
        'annoyance',
        'approval',
        'caring',
        'confusion',
        'curiosity',
        'desire',
        'disappointment',
        'disapproval',
        'disgust',
        'embarrassment',
        'excitement',
        'fear',
        'gratitude',
        'grief',
        'joy',
        'love',
        'nervousness',
        'optimism',
        'pride',
        'realization',
        'relief',
        'remorse',
        'sadness',
        'surprise',
        'neutral']
        system_prompt = f"You are a intelligent helper for sentiment classification. And you should assign the input text with one sentiment from {SENTIMENT_CLASSES}."
        GT_sentiment = [SENTIMENT_CLASSES[int(index)] for index in messages['sentiment_index'].split(',')]
        messages =[{"role":"user", "content": system_prompt}, {"role": "user", "content": f"Text: {messages['text']}"}, {"role": "assistant", "content": GT_sentiment}]
        breakpoint()
        # print(messages)
        prompt_tokens = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
        prompt_len = len(prompt_tokens)
        
        # prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        prompt_response_tokens = tokenizer.apply_chat_template(messages, tokenize=True)
        labels = prompt_response_tokens.copy()
        if not training_args.train_on_prompt:
            labels[:prompt_len] = [label_pad_token_id] * prompt_len
        
        return {
            "input_ids": prompt_response_tokens,
            "attention_mask": [1]*len(prompt_response_tokens),
            "labels": labels,
        }


    # dataset = load_dataset(data_args.dataset_name)
    dataset = load_and_combine_datasets(data_args.dataset_name)
    train_dataset = dataset.shuffle(seed=training_args.seed).map(process_dataset, num_proc=data_args.num_proc)
    # eval_dataset = dataset["test"].shuffle(seed=training_args.seed).map(process_dataset, num_proc=data_args.num_proc)
    # filter too long
    length_train_before = len(train_dataset)
    # length_eval_before =  len(eval_dataset)
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= data_args.max_length, num_proc=data_args.num_proc)
    # eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) <= data_args.max_length, num_proc=data_args.num_proc)
    print_local_main(f"train_dataset_retain: {len(train_dataset) / length_train_before}")
    # print_local_main(f"eval_dataset_retain: {len(eval_dataset) / length_eval_before}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    trainer.train()
    if Accelerator().is_main_process:
        save_name = "best_checkpoint" if training_args.load_best_model_at_end else "final_checkpoint"
        trainer.model.save_pretrained(os.path.join(training_args.output_dir, save_name))
        trainer.tokenizer.save_pretrained(os.path.join(training_args.output_dir, save_name))


if __name__ == "__main__":
    train()
