pip install -r /workspace/gpt_clm/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /workspace/horovod

HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1  pip install -v  .