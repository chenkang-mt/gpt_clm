DATA=${DATADIR:-"/data01"}
python3 save_disk.py \
    --device musa \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --num_train_epochs 200 \
    --block_size 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --checkpointing_steps 1000 \
    --output_dir $DATA/output \
    --cache_dir $DATA/cache \
    --from_checkpoint_meta $DATA/checkpoints \
    --log_dir $DATA/logs \
    --checkpointing_steps 1000 \
    --resume_from_checkpoint 1 \
    --save_dir $DATA