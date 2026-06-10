# legal task order: nli -> mcq -> sqa
# summarization task order: cnn -> arxiv -> mediasum

import os
import argparse
import torch
import sys
import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import gc
from utils.utils import set_random_seed
from utils.llm_data_loader import LLMDataLoader
from utils.customized_trainers import CustomizedTrainer
from model_merging_methods.distill_merging_utils import *
from tqdm import tqdm
import torch.nn.functional as F
import shutil
from dotenv import load_dotenv

load_dotenv()

os.environ['WANDB_DISABLED'] = 'true'

cache_dir = './MergeLM_models'

task_model_mapping_dict = {
    'nli': 'qwen3-1.7b-legal-pretrain-nli',
    'mcq': 'qwen3-1.7b-legal-pretrain-mcq',
    'sqa': 'qwen3-1.7b-legal-pretrain-sqa',
    'cnn': 'qwen3-1.7b-summarization-cnn',
    'arxiv': 'qwen3-1.7b-summarization-arxiv-full',
    'mediasum': 'qwen3-1.7b-summarization-mediasum',
}
finetuned_model_backbone_mapping_dict = {
    'qwen3-1.7b-legal-pretrain-nli': 'qwen3-1.7b-legal-pretrain',
    'qwen3-1.7b-legal-pretrain-mcq': 'qwen3-1.7b-legal-pretrain',
    'qwen3-1.7b-legal-pretrain-sqa': 'qwen3-1.7b-legal-pretrain',
    'qwen3-1.7b-summarization-cnn': 'qwen3-1.7b',
    'qwen3-1.7b-summarization-arxiv-full': 'qwen3-1.7b',
    'qwen3-1.7b-summarization-mediasum': 'qwen3-1.7b',
}
# finetuned_models = ['qwen3-1.7b-legal-pretrain-nli', 'qwen3-1.7b-legal-pretrain-mcq', 'qwen3-1.7b-legal-pretrain-sqa']
finetuned_models = ['qwen3-1.7b-summarization-cnn', 'qwen3-1.7b-summarization-arxiv-full', 'qwen3-1.7b-summarization-mediasum']

parser = argparse.ArgumentParser('Interface for merging LLMs')
parser.add_argument('--do_cnn', action='store_true', help='whether to merge cnn model')
parser.add_argument('--do_arxiv', action='store_true', help='whether to merge arxiv model')
parser.add_argument('--do_mediasum', action='store_true', help='whether to merge mediasum model')
parser.add_argument('--do_nli', action='store_true', help='whether to merge nli model')
parser.add_argument('--do_mcq', action='store_true', help='whether to merge mcq model')
parser.add_argument('--do_sqa', action='store_true', help='whether to merge sqa model')

parser.add_argument('--val_shot', type=int, default=32, help='number of training examples')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')

parser.add_argument('--granularity', type=str, default='elementwise', choices=['taskwise', 'layerwise', 'elementwise'], help='granularity of merging coefficients')

parser.add_argument('--language_model_name', type=str, default='qwen3-1.7b', help='name of the language model')
parser.add_argument('--merging_method_name', type=str, default='sequential_efficient')
parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
parser.add_argument("--tag", type=str, default='test', help="tag for distill merging")
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--layer_save', type=str, default='./save_layers', help='path to save layers in merging')
try:
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
except:
    parser.print_help()
    sys.exit()

def check_gpu():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Cached memory (reserved): {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
        print()


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def train(args, lr, epochs, merged_train_loader, load_model_paths):
    num_layers = 28

    check_gpu()

    # Load a merged embed_tokens layer
    avg_pre_merged_model = load_avg_merged_model_pre_llm(args, merge_coef=0.5)

    check_gpu()

    per_models = []
    for dataset in args.dataset_names:
        finetuned_model = load_single_merged_model_pre_llm(args, dataset)
        per_models.append(finetuned_model)

    check_gpu()

    merged_train_loader = transform_data_loader_prelayer_pertask_llm(
        merged_train_loader, avg_pre_merged_model, per_models, args.device
    )

    # Extract the rotary embedding module and hidden_size so we can compute
    # position_ids / (cos, sin) dynamically for each batch's actual sequence length.
    # This avoids the assumption that every batch is padded to max_length.
    rotary_emb = avg_pre_merged_model.rotary_emb
    hidden_size = avg_pre_merged_model.config.hidden_size

    del avg_pre_merged_model, per_models
    torch.cuda.empty_cache()
    # Embeddings after embed_tokens are now stored in merged_train_loader.

    check_gpu()

    print('HERE START TRAINING!!!')
    for layer_idx in range(num_layers):
        print(f'Training layer {layer_idx}')
        # Load merged_layer (a specific layer without weights), and layers (the same specific layer of finetuned models)
        merged_layer, layers = load_merged_layers_llm(args=args, layer_idx=layer_idx)
        # Turn merged_layer to training mode, and turn all finetuned models' layers to eval mode
        merged_layer.train()
        for layer in layers:
            layer.eval()
        optimizer = torch.optim.Adam(merged_layer.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            total_loss = 0 # total_loss over 1 epoch
            for data in merged_train_loader:
                x = data['data'].to(args.device)
                batch_size = x.shape[0]
                x = x.permute(1, 0, 2, 3).to(torch.bfloat16)  # cast hidden states to bf16

                source_loader = data['source_loader'].to(args.device)

                optimizer.zero_grad()

                # ── Dynamic rotary embeddings ──────────────────────────────────
                # x[0] shape: [batch_size, seq_len, hidden_size]
                # seq_len varies per batch because we use dynamic padding.
                seq_len = x.shape[2]  # dim-2 after permute: [num_views, batch, seq, hidden]
                position_ids = torch.arange(seq_len, device=args.device).unsqueeze(0).expand(batch_size, -1)
                _dummy = torch.zeros(batch_size, seq_len, hidden_size, device=args.device, dtype=torch.bfloat16)
                position_embeddings = rotary_emb(_dummy, position_ids)  # (cos, sin)
                del _dummy

                # Forward through the merged decoder layer; pass position_ids and
                # rotary embeddings (cos, sin) explicitly — Qwen3DecoderLayer requires them.
                # The decoder layer returns a plain Tensor (not a tuple) — use it directly.
                feature = merged_layer.get_merged_model()(
                    x[0],
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                ).reshape(batch_size, -1)

                loss = 0
                idx = source_loader.item()
                with torch.no_grad():
                    # get hidden state of the corresponding finetuned model's layer
                    true_feature = layers[idx](
                        x[1],
                        attention_mask=None,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    ).reshape(batch_size, -1)
                loss += F.mse_loss(feature, true_feature, reduction='none').sum()
                total_loss += loss.detach().clone().cpu().item()
                loss.backward()
                optimizer.step()
            logger.info(f'Layer {layer_idx + 1}/{num_layers}, Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss}')
        print(f'Layer {layer_idx + 1}/{num_layers} finished')

        # Pass output of current layer to get input for the next layer
        merged_train_loader = transform_data_loader_layer_pertask_llm(
            merged_train_loader, merged_layer.get_merged_model(), layers, args.device,
            rotary_emb, hidden_size,
        )
        layer_save_dir = f'{args.layer_save}/{args.dataset_name_combined}/{args.merging_method_name}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}'
        os.makedirs(layer_save_dir, exist_ok=True)
        torch.save(merged_layer.get_merged_model(), f'{layer_save_dir}/layer_{layer_idx}.pt')
        
        del merged_layer, layers
        torch.cuda.empty_cache()

        check_gpu()

    merged_model = load_avg_merged_model_llm(args, merge_coef=0.5)
    for layer_idx in range(num_layers):
        merged_layer = torch.load(f'{layer_save_dir}/layer_{layer_idx}.pt', map_location=args.device, weights_only=False)
        for name, _ in merged_model.model.layers[layer_idx].named_parameters():
            set_attr(merged_model.model.layers[layer_idx], name.split('.'), nn.Parameter(get_attr(merged_layer, name.split('.'))))

    # Delete the dir saving trained layer
    # shutil.rmtree(layer_save_dir)

    return merged_model



if __name__ == '__main__':
    start_time = time.time()
    args.dataset_names = []
    if args.do_nli: args.dataset_names.append('nli')
    if args.do_mcq: args.dataset_names.append('mcq')
    if args.do_sqa: args.dataset_names.append('sqa')
    if args.do_cnn: args.dataset_names.append('cnn')
    if args.do_arxiv: args.dataset_names.append('arxiv')
    if args.do_mediasum: args.dataset_names.append('mediasum')

    args.dataset_name_combined = '_'.join(args.dataset_names)
    args.cache_dir = cache_dir
    args.task_model_mapping_dict = task_model_mapping_dict
    args.finetuned_model_backbone_mapping_dict = finetuned_model_backbone_mapping_dict
    args.finetuned_models = finetuned_models

    set_random_seed(seed=0)

    load_model_paths = []
    for dataset_name in args.dataset_names:
        load_model_paths.append(f"./MergeLM_models/{task_model_mapping_dict[dataset_name]}")

    args.save_merged_model_path = f"./save_merge_models/{args.dataset_name_combined}/{args.merging_method_name}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}"
    os.makedirs(args.save_merged_model_path, exist_ok=True)
    _log_file_handler = logging.FileHandler(f"{args.save_merged_model_path}/train_{args.val_shot}.log")
    _log_file_handler.setLevel(logging.DEBUG)
    _log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_log_file_handler)
    args.load_model_paths_dict = {
        args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir
        )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.add_eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm_data_loader = LLMDataLoader(tokenizer=tokenizer)

    check_gpu()

    model_to_merge = load_pretrained_model(args)

    check_gpu()

    trainers, eval_datasets = [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        train_dataset, test_dataset = llm_data_loader.load_dataset(dataset_name=dataset_name, val_shot=args.val_shot)

        # NOTE: THIS CUSTOMIZED TRAINER IS JUST USED TO GET DATALOADER, NOT TO COMPUTE LOSS OR TRAINING.
        trainer = CustomizedTrainer(
            model=model_to_merge,
            args=TrainingArguments(
                args.save_merged_model_path,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
            ),
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        trainers.append(trainer)
        eval_datasets.append(test_dataset)

    print(llm_data_loader.max_len)

    check_gpu()

    merged_train_loader = merge_data_loaders_from_trainers(trainers)

    for trainer in trainers:
        trainer.model = None
        del trainer
    
    del trainers
    del model_to_merge

    gc.collect()
    torch.cuda.empty_cache()

    check_gpu()

    logger.info(f'********** Run starts. **********')
    logger.info(f'Configuration is {args}')

    check_gpu()

    # NOTE: HERE THE MAIN FUNCTION USE TRAIN FUNCTION DEFINED ABOVE FOR LLM KD, NOT THE CUSTOMIZED TRAINER.
    merged_model = train(args, args.lr, args.epochs, merged_train_loader, load_model_paths)

    end_time = time.time()

    logger.info(f'Run finished in {end_time - start_time} seconds with val shot {args.val_shot}')

    os.makedirs(args.save_merged_model_path, exist_ok=True)

    merged_model.save_pretrained(args.save_merged_model_path)
    tokenizer.save_pretrained(args.save_merged_model_path)
    
    # save eval_datasets
    for dataset_name, eval_dataset in zip(args.dataset_names, eval_datasets):
        indices = eval_dataset.indices
        torch.save(indices, f"{args.save_merged_model_path}/{dataset_name}_indices.pt")
