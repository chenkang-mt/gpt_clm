import json
import os
from threading import Timer
import uuid
import time
import random
import mlflow
import torch

import logging
import time
from accelerate.logging import get_logger
from log import logger

from metrics import push_metrics, build_registry
from metrics import MyHistogram, MySummary, label_wrapper

from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
)

CHECKPOINT_META_NAME = "checkpoint.meta"

def timecost_wrapper(func, world=0, rank=0):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_cost = end_time - start_time

        cost = {func.__name__ : time_cost}

        reg = build_registry()
        s = MyHistogram(f'{func.__name__}_cost_seconds', 'op timecost',
                      label_wrapper(["world", "op_name", "rank"]), registry=reg)
        s.labels(name=func.__name__, op_name=func.__name__, world=world, rank=rank).observe(time_cost)
        push_metrics(reg, rank)

        logger.info(f"time cost statistics: {cost}")
        return result
    
    return wrapper

@timecost_wrapper
def load_latest_ckpt_meta(ckpt: str):
    if len(ckpt) == 0:
        return

    meta = {}
    try:
        with open(ckpt, "r") as input:
            data = input.read()
            if len(data) > 0:
                meta = json.loads(data)
    except Exception as e:
        logger.info(f"load_latest_ckpt_meta catch exception:{e}")

    return meta

@timecost_wrapper
def save_latest_ckpt_meta(ckpt: str, meta: dict):
    if len(meta) == 0:
        return
    
    try:
        file_prefix = os.path.splitext(ckpt)[0]
        data = json.dumps(meta)
        uid = uuid.uuid4().__str__()
        tmp_file = file_prefix + "_" + uid + ".tmp"
        with open(tmp_file, 'w') as output:
            output.write(json.dumps(meta))

        os.rename(tmp_file, ckpt)
    except Exception as e:
        logger.info(f"save_latest_ckpt_meta catch exception:{e}")

    return


@timecost_wrapper
def save_ckpt(prefix:str, model, optimizer, rank, epoch, step):
    try:
        ckpts = os.path.join(prefix, "checkpoints")
        logger.info(f"create checkpoints dirs:{ckpts}")
        os.makedirs(ckpts)
    except Exception as e:
        pass

    cur_date = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    meta = {}
    
    state = f'{cur_date}_rank_{rank}_epoch_{epoch}_step{step}'
    model_file = os.path.join(ckpts, f'{state}.pt')
    meta.update({"model": model_file})
    torch.save(model.state_dict(), model_file)

    optim_file = os.path.join(ckpts, f'{state}.opt')
    torch.save(optimizer.state_dict(), optim_file)
    meta.update({"optimizer": optim_file})

    meta.update({"epoch": epoch})
    meta.update({"step": step})
    meta.update({"rank": rank})

    save_latest_ckpt_meta(os.path.join(prefix, CHECKPOINT_META_NAME), meta)

@timecost_wrapper
def load_ckpt(model, opt, prefix: str):
    meta = load_latest_ckpt_meta(os.path.join(prefix, CHECKPOINT_META_NAME))
    if not meta:
        return  0, 0
    
    try:
        if "model" in meta:
            model.load_state_dict(torch.load(meta["model"]))
        if "optimizer" in meta:
            opt.load_state_dict(torch.load(meta["optimizer"]))
    except Exception as e:
        raise(f"model load exception: {e}")
        
    return  meta["epoch"], meta["step"]



def timecost_stat(func, name, rank, report=True, step=0):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_cost = end_time - start_time

        cost = {func.__name__ : time_cost}
        logger.info(f"time cost statistics: {cost}")
        if report and rank == 0:
            mlflow.log_metric(name, time_cost, step)

        return result
    
    return wrapper



class TimeTicker:
    def __init__(self, name, world: int, rank: int, switch=True, reporter=False, step=0):
        self.start_time = None
        self.end_time = None
        self.time_cost = None
        self.name  = name
        self.rank = rank
        self.world = world
        self.switch = switch
        self.reporter = reporter
        self.step = step

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.time_cost = self.end_time - self.start_time


        if self.switch:
            reg = build_registry()
            s = MyHistogram(f'{self.name}_cost_seconds', 'op timecost', label_wrapper(["op_name", "world", "rank"]), registry=reg)
            s.labels(name=self.name, op_name=self.name, world=self.world, rank=self.rank).observe(self.time_cost)
            push_metrics(reg, self.rank)

            logger.info(f"time cost op:{self.name}, cost:{self.time_cost}, step:{self.step}")
            if self.reporter and self.rank == 0:
                mlflow.log_metric(self.name, self.time_cost, step=self.step)
        


@timecost_wrapper
def cal_model_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += torch.numel(param)

    return total_size



def trace_handler(p):
    logger.info("trace handler enter...")
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1)
    logger.info(output) 

def configure_training_profiler(trace_handler) -> profile:
    profile_schedule = schedule(wait=1, warmup=1, active=1)
    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profile_schedule,
        record_shapes=True,
        on_trace_ready=trace_handler,
        with_flops=True,
        # profiler now provides `flops` attribute
    )

    return profiler
