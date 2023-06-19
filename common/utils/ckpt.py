import json
import os
import time
import uuid
import torch
from common.logger.log import logger
from common.utils.timecost import timecost_wrapper

CHECKPOINT_META_NAME = "checkpoint.meta"

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
