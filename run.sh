ulimit -n 8092
ulimit -c unlimited
rm  -rf /usr/local/lib/libdrm.so
python3 /workspace/gpt_clm/train_hvd_clm.py "$@"