from class_ import MergingCoefficients
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = 'vohuutridung/qwen3-1.7b-legal-pretrain'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

coef = MergingCoefficients(model, num_tasks=3,)
coef.summary()