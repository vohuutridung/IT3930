import torch
from torch.utils.data import DataLoader
from training_loop import prodistill, build_merged_model
from task_vector import compute_all_task_vectors
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import FewShotPipeline, config

tokenizer = AutoTokenizer.from_pretrained("vohuutridung/qwen3-1.7b-legal-pretrain")

pretrained = AutoModelForCausalLM.from_pretrained("vohuutridung/qwen3-1.7b-legal-pretrain", torch_dtype=torch.float16)
ft_nli     = AutoModelForCausalLM.from_pretrained("vohuutridung/qwen3-1.7b-legal-pretrain-nli", torch_dtype=torch.float16)
ft_mcq     = AutoModelForCausalLM.from_pretrained("vohuutridung/qwen3-1.7b-legal-pretrain-mcq", torch_dtype=torch.float16)
ft_sqa     = AutoModelForCausalLM.from_pretrained("vohuutridung/qwen3-1.7b-legal-pretrain-sqa", torch_dtype=torch.float16)

pipeline = FewShotPipeline(config)
hf_dataset = pipeline.build()

def make_loader(task_id: int, batch_size: int = 1):
    task_ds = hf_dataset.filter(lambda x: x['task_id'] == task_id)
    task_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return DataLoader(task_ds, batch_size=batch_size, shuffle=True)

# task_id 0 = nli, 1 = mcq, 2 = sqa  (matches config['tasks'] order)
loader_nli = make_loader(0)
loader_mcq = make_loader(1)
loader_sqa = make_loader(2)

# loaders order must match finetuned_models order
loaders = [loader_nli, loader_mcq, loader_sqa]

# Build layer names for the pretrained model
layer_names = []
layer_names.append("model.embed_tokens")
num_layers = pretrained.config.num_hidden_layers
for i in range(num_layers):
    layer_names.append(f"model.layers.{i}")

coefficients = prodistill(
    pretrained_model=pretrained,
    finetuned_models=[ft_nli, ft_mcq, ft_sqa],
    dataloaders=loaders,
    layer_names=layer_names,
    num_epochs=20,        # 20 epochs per layer (paper default)
    lr=0.1,               # paper default
    verbose=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
print('Training completed.')

task_vectors = compute_all_task_vectors(pretrained, [ft_nli, ft_mcq, ft_sqa])
merged_model = build_merged_model(pretrained, task_vectors, coefficients)

merged_model.eval()
inputs = tokenizer("Your input here", return_tensors="pt")
output = merged_model(**inputs)