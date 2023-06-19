from typing import Any, Iterable, Optional, Sequence, Union
from prometheus_client import  Gauge, Counter, Histogram, Summary
from prometheus_client import instance_ip_grouping_key, pushadd_to_gateway, delete_from_gateway, CollectorRegistry
import time
import os
import socket
import uuid
from threading import Timer
from prometheus_client.registry import REGISTRY, CollectorRegistry
#from log import logger


kPrometheusUrl = ""
kServiceName  = ""
kEnv = ""

def init_prometheus(url: str, service: str, env: str):
    if len(url) <= 0:
        raise(f"illegal gateway url{url}")
    
    global kPrometheusUrl
    global kServiceName
    global kEnv
    kPrometheusUrl = url
    kServiceName = service
    kEnv = env

def build_registry(): 
    registry = CollectorRegistry()
    return registry

def push_metrics(registry: CollectorRegistry, rank = 0):
    try:
        grpkey = {"rank": rank}
        pushadd_to_gateway(kPrometheusUrl, kServiceName, registry, grouping_key=grpkey)
    except Exception as e:
        print(f"exception {e}")
        #logger.warning(f"failed to push metrics, cause of {e}")

def delete_metrics():
    try:
        delete_from_gateway(kPrometheusUrl, kServiceName)
    except Exception as e:
        print(f"exception {e}")


def label_wrapper(labels: list = []):
    labels.append("service")
    labels.append("instance")
    labels.append("env")
    labels.append("job")
    labels.append("name")

    return labels


def label_kw_wrapper(func):
    def wrapper(*args, **kwargs):
        hostname = socket.gethostname()
        kwargs.update({
            "service": kServiceName, 
            "instance":  socket.gethostname(),
            "env": kEnv,
            "job": kServiceName,
        })
        u = func(*args, **kwargs)
        return u

    return wrapper


class MyCounter(Counter):
    @label_kw_wrapper
    def labels(self: Counter, *labelvalues: Any, **labelkwargs: Any) -> Counter:
        return super(Counter, self).labels(*labelvalues, **labelkwargs)

class MyHistogram(Histogram):
    @label_kw_wrapper
    def labels(self: Histogram, *labelvalues: Any, **labelkwargs: Any) -> Histogram:
        return super(MyHistogram, self).labels(*labelvalues, **labelkwargs)


class MySummary(Summary):
    @label_kw_wrapper
    def labels(self: Summary, *labelvalues: Any, **labelkwargs: Any) -> Summary:
        return super(MySummary, self).labels(*labelvalues, **labelkwargs)
    

class MyGauge(Gauge):
    @label_kw_wrapper
    def labels(self: Gauge, *labelvalues: Any, **labelkwargs: Any) -> Gauge:
        return super(MyGauge, self).labels(*labelvalues, **labelkwargs)

def report_loss(lr: float, loss: float, stp: int , rk: int):
    reg = build_registry()
    labels = label_wrapper(["rank"])
    g1 = MyGauge("loss", "loss value", labels, registry=reg)
    g1.labels(name=g1._name, rank=rk).set(loss)
    g2 = MyGauge("step", "step value", labels, registry=reg)
    g2.labels(name=g2._name, rank=rk).set(stp)
    g3 = MyGauge("lr", "learning rate", labels, registry=reg)
    g3.labels(name=g3._name, rank=rk).set(lr)

    push_metrics(reg, rk)  

def report_basics(np, batch, seq_len, rk, model_size):
    reg = build_registry()
    labels = label_wrapper(["rank"])

    g1 = MyGauge("batch", "batch size", labels, registry=reg)
    g1.labels(name=g1._name, rank=rk).set(batch)
    g2 = MyGauge("np", "num of procs", labels, registry=reg)
    g2.labels(name=g2._name, rank=rk).set(np)
    g3 = MyGauge("seq_len", "seq length", labels, registry=reg)
    g3.labels(name=g3._name, rank=rk).set(seq_len)

    g4 = MyGauge("model_size", "model size", labels, registry=reg)
    g4.labels(name=g4._name, rank=rk).set(model_size)

    push_metrics(reg, rk)

    #start_timer(np, batch, seq_len, rk)

def start_timer(np, batch, seq_len, rk, model_size):
    def func():
        report_basics(np, batch, seq_len, rk, model_size)
    t = Timer(10, func)
    t.start()



if __name__ == '__main__':
    init_prometheus("http://192.168.41.156:9091", "musa-hvd-gpt2", "test")
    delete_metrics()
    import sys
    sys.exit(0)
    reg = build_registry()
    labels = label_wrapper(["op_name", "rank"])
    # print(labels)
    s = MyHistogram('hectorgao_seconds_stat', 'op timecost', labels, registry=reg)
    s.labels(name="test_op", op_name="test_op", rank=0).observe(70)

    push_metrics(reg)