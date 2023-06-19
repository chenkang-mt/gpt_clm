batch_size=4
device="musa"
seq_len=1024
np=1

for (( i=1; i<=$#; i++ )); do
    eval opt='$'{${i}}
    case ${opt} in
        -b)
            let "i++"
            eval batch_size='$'{${i}}
            ;;
        -d)
            let "i++"
            eval device='$'{${i}}
            ;;
        -s)
            let "i++"
            eval seq_len='$'{${i}}
            ;;
        -p)
            let "i++"
            eval np='$'{${i}}
            ;;
        *)
            echo "Error : invalid opt ${opt}"
            exit 1
    esac
done

ulimit -n 8092
ulimit -c unlimited


DATA=${DATADIR:-"/data01"}

horovodrun -np $np -H 127.0.0.1:$np python3 train_hvd_clm.py \
    --device $device \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --num_train_epochs 1 \
    --block_size $seq_len \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 2 \
    --output_dir $DATA/output \
    --cache_dir $DATA/cache \
    --max_train_steps 100 \
    --from_checkpoint_meta $DATA/checkpoints \
    --log_dir $DATA/logs \
    --resume_from_checkpoint 0 \
    --save_dir $DATA \
    --enable_prof 0