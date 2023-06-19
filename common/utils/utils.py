import torch
from common.utils.ckpt import timecost_wrapper



@timecost_wrapper
def cal_model_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += torch.numel(param)

    return total_size



