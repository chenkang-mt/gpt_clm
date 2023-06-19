import time

import mlflow
from common.logger.log import logger
from common.telemetry.metrics import MyHistogram, build_registry, label_wrapper, push_metrics

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