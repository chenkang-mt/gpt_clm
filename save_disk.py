
import os
from datasets import load_dataset, load_from_disk
import config
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

args = config.parse_args()
if __name__ == '__main__':
    
    dataset_dir = os.path.join(args.save_dir, args.dataset_config_name)
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    raw_datasets.save_to_disk(dataset_dir)


    tokenizer_dir = os.path.join(args.save_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, use_fast=not args.use_slow_tokenizer)
    tokenizer.save_pretrained(tokenizer_dir)

    model_dir = os.path.join(args.save_dir, "models")
    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
        model.save_pretrained(model_dir)

    
