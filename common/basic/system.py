import os
from cpuinfo import get_cpu_info

def obtain_system_info():
    res = {}
    try:
        os_info = os.uname()
        res["sysname"] = os_info.sysname
        res["kernel"] = os_info.release
        res["os"] = os_info.nodename
    except Exception as e:
        pass

    return res


def obtain_cpu_info():
    info = get_cpu_info()
    res = {}
    try:
        res["arch"] = info["arch"]
        res["count"] = info["count"]
        res["bits"] = info["bits"]
        res["vendor_id_raw"] = info["vendor_id_raw"]
        res["brand_raw"] = info["brand_raw"]
        res["hz_actual_friendly"] = info["hz_actual_friendly"]
    except Exception as e:
        pass

    return res