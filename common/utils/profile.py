from common.logger.log import logger
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
)


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