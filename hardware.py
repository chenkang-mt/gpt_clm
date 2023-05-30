import os
from subprocess import PIPE, Popen


class MTGPU:
    def __init__(self, driver, name, num, mtbios):
        self.driver = driver
        self.name = name
        self.num = num
        self.mtbios = mtbios


def getMTGPUs():
    mthreads_gmi = "mthreads-gmi"

    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen([mthreads_gmi,
                   "-q -i 0"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')

    lines = output.split(os.linesep)
    # print(lines)
    MTGPUs = []
    for line in lines:
        # print(line)
        kvs = line.split(' : ')
        if len(kvs) != 2:
            continue
        if kvs[0].strip().startswith("Driver Version"):
            driver = kvs[1].strip()
        if kvs[0].strip().startswith("Attached GPUs"):
            gpus_num = int(kvs[1].strip())
        if kvs[0].strip().startswith("Product Name"):
            name = kvs[1].strip()
        if kvs[0].strip().startswith("GPU UUID"):
            uuid = kvs[1].strip()
        if kvs[0].strip().startswith("MTBios Version"):
            mtbios = kvs[1].strip()

    MTGPUs.append(MTGPU(driver, name, gpus_num, mtbios))

    return MTGPUs

def obtain_mtgpu_info():
    gpu_info = {}
    GPUs = getMTGPUs()
    if len(GPUs) == 0:
        return gpu_info

    gpu_info["name"] = GPUs[0].name
    gpu_info['driver'] = GPUs[0].driver
    gpu_info['mtbios'] = GPUs[0].mtbios
    gpu_info['num'] = GPUs[0].num

    return gpu_info