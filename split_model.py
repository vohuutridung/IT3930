# Download model from Hugging Face and split into layers

from transformers import AutoModelForCausalLM
import argparse
import os
import torch
import json
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser('Interface for splitting model')
parser.add_argument('--model_name', type=str, required=True)
args = parser.parse_args()

# Download the whole model repo to local dir
model_name = 'vohuutridung/' + args.model_name
snapshot_download(model_name, local_dir=f'./MergeLM_models/{args.model_name}')

# Load the full model
model_path = f'./MergeLM_models/{args.model_name}'
split_dir = os.path.join(model_path, 'split')
os.makedirs(split_dir, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Prepare the weight map
weight_map = {}

# Save embedding layer
torch.save(model.model.embed_tokens, os.path.join(split_dir, 'model_embed_tokens.pt'))
weight_map['model.embed_tokens'] = 'model_embed_tokens.pt'

# Iterate through layers to save layers
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]

    torch.save(layer, os.path.join(split_dir, f'model_layer_{i}.pt'))
    weight_map[f'model.layers.{i}'] = f'model_layer_{i}.pt'

# Save norm and lm_head
torch.save(model.model.norm, os.path.join(split_dir, 'model_norm.pt'))
weight_map['model.norm'] = 'model_norm.pt'

torch.save(model.lm_head, os.path.join(split_dir, 'model_lm_head.pt'))
weight_map['lm_head'] = 'model_lm_head.pt'

# Save the index file 
with open(os.path.join(split_dir, 'model_index.json'), 'w') as index_file:
    json.dump(weight_map, index_file, indent=4)

del model
torch.cuda.empty_cache()

print('Model structure and weights have been saved, and the index file has been generated.')
