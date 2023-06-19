import os
import time
from accelerate.logging import get_logger
import logging

logger = get_logger(__name__)

def init_logs(log_dir, rank):
    try:
        os.makedirs(log_dir)
    except Exception as e:
        pass

    cur_date  = time.strftime("%Y-%m-%d-%H", time.localtime())
    log_file = os.path.join(log_dir, f"{cur_date}_train_{rank}.log")
    logging.basicConfig(
        #filename= log_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    return log_file

