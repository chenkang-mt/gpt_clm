#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import time
import datasets
import torch_musa
import torch

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from common.utils.utils import TimeTicker, timecost_stat, cal_model_size, timecost_wrapper, load_ckpt, save_ckpt 
from  common.logger.log import init_logs, logger
from common.utils.utils import configure_training_profiler, trace_handler
import config 
from  common.telemetry.metrics import init_prometheus, push_metrics, start_timer, report_loss

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0")


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def rank():
    return dist.get_rank()

def world_size():
    return dist.get_world_size()

def setup():
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("gloo")
    torch_musa.set_device(rank() % torch.musa.device_count())

def cleanup():
    dist.destroy_process_group()


    
@timecost_wrapper(world_size(), rank())
def create_dataset(tokenizer):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )

    logger.info(f"raw datasets:{raw_datasets.keys()}")

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    return train_dataset
    


@timecost_wrapper(world_size(), rank())
def load_tokenizer():
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    if args.save_dir:
        t_dir = os.path.join(args.save_dir, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(t_dir, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    return tokenizer

@timecost_wrapper(world_size(), rank())
def build_model():
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.save_dir:
        m_dir = os.path.join(args.save_dir, "models")
        config = AutoConfig.from_pretrained(m_dir, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.save_dir:
        model = AutoModelForCausalLM.from_config(
            config=config,
        )
    elif args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    device = torch.device(args.device)
    model = model.to(device)

    ddp_model = DDP(model, device_ids=[rank()])

    return ddp_model

def is_master():
    if rank() == 0:
        return True
    return False

def prof_switch():
    if args.enable_prof and is_master():
        return True
    
    return False

def main():

    device = torch.device(args.device)
    log_file = init_logs(args.log_dir, rank())
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    init_prometheus(args.gateway, "musa-ddp-gpt2", "test")
    


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if is_master():
        mlflow.log_params(vars(args))

    tokenizer = load_tokenizer()
    train_dataset = create_dataset(tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size(), 
        rank=rank())
    
    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=default_data_collator, 
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler, 
        drop_last=True,
        num_workers=1
    )

    start_epoch = 0
    start_step = 0
    
    model = build_model()


        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
      {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
      },
      {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
      },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    if args.resume_from_checkpoint:
        start_epoch, start_step = load_ckpt(model, optimizer, args.output_dir)

    model_size = cal_model_size(model)
    start_timer(world_size(), args.per_device_train_batch_size, args.block_size, rank(), model_size)
    logger.info(f"succeed  to load model, and model size is {model_size}")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    # )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    
    # model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states


    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Train num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    resume_step = start_step
    starting_epoch = start_epoch
    completed_steps = resume_step
    checkpointing_steps = int(args.checkpointing_steps)

    # Potentially load in the weights and states from a previous save
    global_step = 0


    for epoch in range(starting_epoch, args.num_train_epochs):
        start_train = time.time()
        model.train()
        total_loss = 0
        total_sample = 0
        io_start = time.time()
    
        prof=None
        if prof_switch():
            prof = configure_training_profiler(trace_handler)
            prof.start()
          
        for step, batch in enumerate(train_dataloader) :
            # We need to skip steps until we reach the resumed step
            if  epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    pass

            cond =  step % 5
            if not cond and is_master():
                mlflow.log_metric("data_loader", time.time() -  io_start, completed_steps)

            with TimeTicker("step_train",world_size(), rank(), True, True, completed_steps) as t:
                batch['input_ids'] = batch['input_ids'].to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                batch['labels'] = batch['labels'].to(device)
            
                with TimeTicker("forward", True, True, completed_steps) as t:
                    outputs = model(**batch)
                    # torch.musa.synchronize()

                loss = outputs.loss
                total_loss += loss.detach().float()
                report_loss(args.learning_rate, loss.item(), completed_steps, rank())

                with TimeTicker("backward", True, True, completed_steps) as t:
                    loss.backward()
                    # torch.musa.synchronize()
                '''   
                has_nan = False
                for n, p in model.named_parameters():
                    try:
                        if torch.any(torch.isnan(p.grad)):
                            has_nan = True
                            break
                        if torch.any(torch.abs(p.grad) > 1):
                            has_nan = True
                            break
                    except:
                        pass

                if has_nan:
                    logger.info('gradient has nan, drop this step')
                    optimizer.zero_grad()
                    io_start = time.time()
                    continue
                '''

                with TimeTicker("optimizer", world_size(), rank(), True, True, completed_steps) as t:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                if prof and step % 5 == 0:
                    prof.step()

            total_sample +=  len(batch['input_ids'])
            train_time = time.time() - start_train
            train_fps =  total_sample / train_time  * args.block_size
            io_start = time.time()


            logger.info(f"epoch {epoch}, step {completed_steps}, train_loss: {loss}  train_time: {train_time:.3f}  train_fps: {train_fps:.3f}")

            if is_master(): 
                mlflow.log_metric("num_procs", world_size())
                mlflow.log_metric("seq_len", args.block_size)
                mlflow.log_metric("batch_size", args.per_device_train_batch_size)
                mlflow.log_metric("loss", loss, completed_steps)
                mlflow.log_metric("train_qps", train_fps, completed_steps)
                
                #completed_steps = hvd.allreduce(torch.tensor(completed_steps), name='completed_steps').item()
            if completed_steps % checkpointing_steps == 0 and  completed_steps:
                if rank() == 0:
                    save_ckpt(args.output_dir, model, optimizer, rank(), epoch, completed_steps)
        
            completed_steps += 1
            if completed_steps >= args.max_train_steps:
                break
        if prof:
            prof.stop()
            logger(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        

        train_time = time.time() - start_train
        train_fps = len(train_dataloader.dataset) / train_time
        logger.info(f"epoch {epoch}: train_loss: {total_loss.item() / len(train_dataloader):.3f}  train_time: {train_time:.3f}  train_fps: {train_fps:.3f}")


args = config.parse_args()


if __name__ == "__main__":
    setup()

    import sys
    import mlflow
    import common.basic.system as system
    import common.basic.hardware as hardware
    
    if rank() == 0:
        mtags = {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cpu": system.obtain_cpu_info(),
            "gpu": hardware.obtain_mtgpu_info(),
        }
        
        name = f"gpt2-ddp-k256-np{world_size()}-batch{args.per_device_train_batch_size}-seq{args.block_size} "
        with mlflow.start_run(run_name=name, tags=mtags) as run:
            main()
    else:
        main()