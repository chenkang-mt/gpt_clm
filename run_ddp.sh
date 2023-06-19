DATA=${DATADIR:-"/data01"}
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 ddp_train_clm.py \
    --device musa \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --num_train_epochs 200 \
    --block_size 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --checkpointing_steps 10000 \
    --output_dir $DATA/output \
    --cache_dir $DATA/cache \
    --from_checkpoint_meta $DATA/checkpoints \
    --log_dir $DATA/logs \
    --resume_from_checkpoint 1 \
    --save_dir $DATA \
    --gateway "http://192.168.41.156:9091"