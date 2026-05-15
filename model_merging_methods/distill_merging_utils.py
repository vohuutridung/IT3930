from numpy import isin
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from torch.utils.data import Dataset, DataLoader
import random
import json
import torch.nn as nn
import copy

# Single source-of-truth for floating-point dtype used everywhere in this file.
# bfloat16 is preferred: it has the same dynamic range as float32 but uses half
# the memory, and Ampere+ GPUs execute bf16 ops natively (no accuracy loss vs tf32).
COMPUTE_DTYPE = torch.bfloat16


def check_gpu():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Cached memory (reserved): {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
        print()


def remove_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def set_attr(obj, names, value):
    """
    Set an attribute of an object recursively
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        set_attr(getattr(obj, names[0]), names[1:], value)

def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def load_pretrained_model(args):
    if args.language_model_name == 'qwen3-1.7b-legal-pretrain' or args.language_model_name == 'qwen3-1.7b':
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(args.cache_dir, args.language_model_name),
            device_map=args.device,
            torch_dtype=COMPUTE_DTYPE,
        )
        remove_grad(pretrained_model)
    return pretrained_model


def get_weight_map_llm(model_name, args):
    model_path = os.path.join(args.cache_dir, model_name, 'split')
    weight_map = json.load(open(os.path.join(model_path, 'model_index.json')))
    return weight_map


def load_part_model(args, module_name, model_name):
    """
    Load a specific module of the model
    """
    weight_map = get_weight_map_llm(model_name, args)
    model_path = os.path.join(args.cache_dir, model_name, 'split')
    weight_path = os.path.join(model_path, weight_map[module_name])
    model = torch.load(weight_path, weights_only=False).to(device=args.device, dtype=COMPUTE_DTYPE)
    remove_grad(model)
    return model


def load_avg_merged_model_pre_llm(args, merge_coef=0.5):
    """
    Task Arithmetic in embed_tokens layer, it's simple linear combination of embed_tokens layer with merge_coef = 0.5
    Return only the merged embed_tokens layer
    """
    pre_model = load_pretrained_model(args).model
    check_gpu()
    del pre_model.norm, pre_model.layers # keep only embed_tokens layer
    check_gpu()

    new_state_dict = {}

    # Iterate through each finetuned model
    for dataset in args.dataset_names:
        # Load only the embed_tokens layer
        model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset])
        # Compute the task vector between pretrained model and the specific finetuned model: lambda_i = sum((W_i - W_0) * merge_coef))
        for name, param in model.named_parameters():
            new_param = (dict(model.named_parameters())[name] - dict(pre_model.named_parameters())[f'embed_tokens.{name}']) * merge_coef + dict(pre_model.named_parameters())[f'embed_tokens.{name}']
            if new_state_dict.get(f'embed_tokens.{name}') is None:
                new_state_dict[f'embed_tokens.{name}'] = new_param
            else:
                new_state_dict[f'embed_tokens.{name}'] += new_param
        del model
        torch.cuda.empty_cache()
    
    # Add the merged embed_tokens layer to the pretrained model: W_0 + sum(lambda_i)
    for name, value in new_state_dict.items():
        set_attr(pre_model, name.split('.'), nn.Parameter(value + dict(pre_model.named_parameters())[name], requires_grad=False))

    return pre_model


def load_single_merged_model_pre_llm(args, dataset):
    pre_model = load_pretrained_model(args).model
    check_gpu()
    del pre_model.norm, pre_model.layers
    check_gpu()

    new_state_dict = {}

    model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset])
    # Compute the task vector between pretrained model and the specific finetuned model: lambda_i = W_i - W_0
    for name, param in model.named_parameters():
        new_param = (dict(model.named_parameters())[name] - dict(pre_model.named_parameters())[f'embed_tokens.{name}']) 
        if new_state_dict.get(f'embed_tokens.{name}') is None:
            new_state_dict[f'embed_tokens.{name}'] = new_param
        else:
            new_state_dict[f'embed_tokens.{name}'] += new_param
    del model
    torch.cuda.empty_cache()

    # W0 + lambda_i = W0 + (W_i - W_0) = W_i -> reconstruct a specific finetuned model, and of course this's just embed_tokens layer
    for name, value in new_state_dict.items():
        set_attr(pre_model, name.split('.'), nn.Parameter(value + dict(pre_model.named_parameters())[name], requires_grad=False))

    return pre_model


def transformed_data_collate_fn(batch):
    data = batch[0][0]
    source_loaders = batch[0][1]
    if len(batch[0]) > 2:
        attention_mask = batch[0][2]
    else:
        return {'data': data, 'source_loader': source_loaders}
    return {'data': data, 'source_loader': source_loaders, 'attention_mask': attention_mask}


def transform_data_loader_prelayer_pertask_llm(data_loader, merged_model, models, device, num_workers=0, shuffle=True, batch_size=1):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            print(x['input_ids'].shape)
            source_loader = data['source_loader']

            inputs = []

            # Call embed_tokens directly — the output shape is [batch_size, seq_length, embedding_dim]
            merged_embed = merged_model.embed_tokens(x['input_ids']).to(COMPUTE_DTYPE)
            inputs.append(merged_embed)

            model = models[source_loader.item()]
            task_embed = model.embed_tokens(x['input_ids']).to(COMPUTE_DTYPE)
            inputs.append(task_embed)

            # shape: [num_tasks+1, batch_size, seq_length, embedding_dim] -> [batch_size, num_tasks+1, seq_length, embedding_dim]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()

            # batch size = 1
            transformed_data.append((inputs, source_loader))

    new_dataset = TransformedDataDataset(transformed_data)
    new_dataloader = DataLoader(
        new_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=transformed_data_collate_fn,
        num_workers=num_workers,
    )

    return new_dataloader


def transform_data_loader_layer_pertask_llm(data_loader, merged_model, models, device, rotary_emb, hidden_size):
    """
    Pass the hidden states (output of the previous layer) through the current decoder layer
    to produce inputs for the next layer.

    Position IDs and rotary (cos, sin) embeddings are computed dynamically from the actual
    sequence length of each batch, so this function is compatible with dynamic padding
    (no assumption that all batches share the same sequence length).
    """
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            # Reshape x from [batch_size, num_tasks+1, seq_len, embedding_dim]
            # to [num_tasks+1, batch_size, seq_len, embedding_dim]
            x = data['data'].to(device)
            x = x.permute(1, 0, 2, 3)

            source_loader = data['source_loader']

            # ── Dynamic rotary embeddings ──────────────────────────────────────
            batch_size, seq_len = x.shape[1], x.shape[2]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
            # _dummy must be COMPUTE_DTYPE so rotary_emb outputs cos/sin in the same dtype.
            _dummy = torch.zeros(batch_size, seq_len, hidden_size, device=device, dtype=COMPUTE_DTYPE)
            position_embeddings = rotary_emb(_dummy, position_ids)  # (cos, sin) in COMPUTE_DTYPE
            del _dummy

            # Ensure hidden states are in COMPUTE_DTYPE before feeding into layers.
            x = x.to(COMPUTE_DTYPE)

            # inputs for the next layer
            inputs = []

            # Forward through the merged decoder layer
            output = merged_model(
                x[0],
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            inputs.append(output)

            idx = source_loader.item()
            model = models[idx]  # get the corresponding finetuned model's layer
            output = model(
                x[1],
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            inputs.append(output)

            # [num_tasks, batch_size, seq_len, embedding_dim] -> [batch_size, num_tasks, seq_len, embedding_dim]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()

            # batch size = 1
            transformed_data.append((inputs, source_loader))

    new_dataset = TransformedDataDataset(transformed_data)
    new_dataloader = DataLoader(
        new_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=transformed_data_collate_fn,
        num_workers=0,
    )

    return new_dataloader


def load_avg_merged_model_llm(args, merge_coef=0.5):
    """
    Reconstruct complete merged model after training each layer
    """
    pre_model = load_pretrained_model(args)

    modules = ['model.embed_tokens.', 'model.norm.', 'lm_head.']
    for i in range(28):
        modules.append(f'model.layers.{i}.')

    for module in modules:
        for name, param in pre_model.named_parameters():
            if module not in name:
                continue
            value = dict(pre_model.named_parameters())[name].clone()
            for dataset in args.dataset_names:
                model = load_part_model(args, module[:-1], args.task_model_mapping_dict[dataset])
                value += (dict(model.named_parameters())[name[len(module):]] - dict(pre_model.named_parameters())[name]) * merge_coef

                del model
                torch.cuda.empty_cache()
            
            set_attr(pre_model, name.split('.'), nn.Parameter(value, requires_grad=False))
            del value

    return pre_model


def load_merged_layers_llm(args, layer_idx):
    # Load specific layer of pretrained model
    layer_pretrained = load_part_model(args, f'model.layers.{layer_idx}', args.language_model_name)

    # Load specific layer of finetuned models
    layers = []
    for dataset in args.dataset_names:
        layer = load_part_model(args, f'model.layers.{layer_idx}', args.task_model_mapping_dict[dataset])
        layers.append(layer)
    
    merged_layers = MergedModel(layer_pretrained, layers, 'elementwise')
    # merged_layers is a model structure without weights, layers are the same specific layer of finetuned models
    return merged_layers, layers


def make_functional(model):
    original_params = tuple(model.parameters())
    names = []
    for name, param in list(model.named_parameters()):
        del_attr(model, name.split('.'))
        names.append(name)
    return original_params, names


def load_weights(model, names, params):
    for name, param in zip(names, params):
        set_attr(model, name.split('.'), param)


# Main class for init training coefficients
class MergedModel(nn.Module):
    def __init__(self, pretrained_model, models, granularity):
        super(MergedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.models = models
        self.granularity = granularity

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.alphas = nn.ParameterList()
        for model in self.models:
            alpha = nn.ParameterList()
            # 1 alpha for 1 task
            if self.granularity == 'taskwise':
                # Use COMPUTE_DTYPE so alpha * delta arithmetic stays in bf16.
                alpha.append(nn.Parameter(torch.tensor(0.5, dtype=COMPUTE_DTYPE), requires_grad=True))
            # 1 alpha for 1 model parameter
            elif self.granularity == 'layerwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(torch.tensor(0.5, dtype=COMPUTE_DTYPE), requires_grad=True))
            # 1 alpha for each element in a model parameter's tensor
            elif self.granularity == 'elementwise':
                for param in model.parameters():
                    # ones_like inherits the param dtype (already COMPUTE_DTYPE after load_part_model).
                    alpha.append(nn.Parameter(torch.ones_like(param, dtype=COMPUTE_DTYPE) * 0.5, requires_grad=True))
            else:
                raise NotImplementedError(f'Invalid granularity: {self.granularity}')
            self.alphas.append(alpha)

        # Remove parameters from model and keep only structure,
        # enabling dynamic weight injection (for model merging with learnable alphas).
        self.merged_model = copy.deepcopy(self.pretrained_model)
        _, self.names = make_functional(self.merged_model)

    def get_merged_model(self):
        """
        Compute merged weights: pretrained + weighted sum of task deltas, then load them into the functional model.
        """
        merged_param = []
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            # Accumulate in COMPUTE_DTYPE to prevent silent float32 promotion
            # when alpha (bf16 scalar/tensor) multiplies the weight delta.
            param = torch.zeros_like(pretrained_param, dtype=COMPUTE_DTYPE)
            pretrained_bf16 = pretrained_param.to(COMPUTE_DTYPE)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                task_param = dict(self.models[k].named_parameters())[name].to(COMPUTE_DTYPE)
                param += alpha * (task_param - pretrained_bf16)
            param += pretrained_bf16
            merged_param.append(param)
        
        load_weights(self.merged_model, self.names, merged_param)

        return self.merged_model

    def get_named_parameters(self):
        """
        Compute merged weights: pretrained + weighted sum of task deltas, them return mapping of name - parameter
        """
        merged_param = {}
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param, dtype=COMPUTE_DTYPE)
            pretrained_bf16 = pretrained_param.to(COMPUTE_DTYPE)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                task_param = dict(self.models[k].named_parameters())[name].to(COMPUTE_DTYPE)
                param += alpha * (task_param - pretrained_bf16)
            param += pretrained_bf16
            merged_param[name] = param

        return merged_param

    def forward(self, x):
        merged_model = self.get_merged_model()
        if isinstance(x, dict):
            return merged_model(**x)
        else:
            return merged_model(x)
    
    def turn_on_layer(self, layer_idx):
        """
        Turn on gradient of a specific layer
        """
        layer_name = f'layer.{layer_idx}'
        assert self.granularity in ['layerwise', 'elementwise']
        for idx, (name, _) in enumerate(self.pretrained_model.named_parameters()):
            for k in range(len(self.models)):
                alpha = self.alphas[k][idx]
                if layer_name in name:
                    alpha.requires_grad = True
                else:
                    alpha.requires_grad = False


class TransformedDataDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    

class LabeledDataset(Dataset):
    """
    Returns a tuple of (sample, dataset_idx)
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_indices = []
        for i, dataset in enumerate(datasets):
            self.dataset_indices.extend(
                [(i, idx) for idx in range(len(dataset))]
            )
        random.shuffle(self.dataset_indices)
    
    def __len__(self):
        return len(self.dataset_indices)
    
    def __getitem__(self, index):
        dataset_idx, sample_idx = self.dataset_indices[index]
        sample = self.datasets[dataset_idx][sample_idx]
        return sample, dataset_idx


def custom_collate_fn(batch):
    # Custom collate function to handle varying input sizes
    data = [item[0] for item in batch]
    source_loader = torch.tensor([item[1] for item in batch])
    return {
        'data': data,
        'source_loader': source_loader,
    }


def merge_data_loaders_from_trainers(trainers, batch_size=1, num_workers=0):
    # Extract datasets from the data loaders
    datasets = []
    for trainer in trainers:
        dataloader = trainer.get_train_dataloader()
        dataset = []
        for item in dataloader:
            dataset.append(trainer._prepare_inputs(item))
        datasets.append(dataset)
    
    # Create a merged dataset
    merged_dataset = LabeledDataset(datasets)

    # Create a new data loader from the merged dataset
    merged_loader = DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )

    return merged_loader