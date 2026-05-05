import pandas as pd
from transformers import AutoTokenizer
import json
import datasets
import numpy as np
from torch.utils.data import Subset
from dotenv import load_dotenv
import os

load_dotenv()

class LLMDataLoader:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.nli_path = os.getenv('task1_data')
        self.mcq_path = os.getenv('task2_data')
        self.sqa_path = os.getenv('task3_data')
        self.max_len = 0

    def encode(self, examples: dict, max_seq_length: int = 512):
        # Here i set max_length = tokenizer.model_max_length in summarization tasks
        inputs = {}
        ins_token = self.tokenizer(
            examples['instruction'],
            max_length=131072,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        inputs['input_ids'] = ins_token['input_ids']
        inputs['attention_mask'] = ins_token['attention_mask']
        self.max_len = max(self.max_len, inputs['attention_mask'][0].sum().item())

        return inputs

    def load_dataset(self, dataset_name: str, max_seq_length: int = 256, val_shot: int = 64):
        # train num is 64, others is test
        if dataset_name == 'nli':
            data_df = pd.read_json(self.nli_path, lines=True)
            data_df['instruction'] = data_df.apply(format_nli, axis=1)
        elif dataset_name == 'mcq':
            data_df = pd.read_json(self.mcq_path, lines=True)
            data_df['instruction'] = data_df.apply(format_mcq, axis=1)
        elif dataset_name == 'sqa':
            data_df = pd.read_json(self.sqa_path, lines=True)
            data_df['instruction'] = data_df.apply(format_sqa, axis=1)
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')

        dataset = datasets.Dataset.from_pandas(data_df)
        dataset = dataset.map(self.encode, batched=True)

        permuted_indices = np.random.RandomState(seed=0).permutation(len(dataset)).tolist()
        num_train_data = val_shot
        train_dataset = Subset(dataset=dataset, indices=permuted_indices[:num_train_data])
        test_dataset = Subset(dataset=dataset, indices=permuted_indices[num_train_data:])
        return train_dataset, test_dataset
        
        



def format_nli(example):
    return (f"""
    Dưới đây là các câu hỏi suy luận pháp lý (có đáp án) về suy luận ngôn ngữ tự nhiên. Hãy phân loại "Có" hoặc "Không".

    Tài liệu pháp luật:
    {example['legal_document']}

    Câu hỏi cụ thể:
    {example['specific_question']}

    Câu hỏi: Điều luật được cung cấp có thể dùng để trả lời câu hỏi trên hay không?
    Có
    Không
    """.strip()
    )

def format_mcq(example):
    choices_text = '\n'.join(example['choices'])

    return ( f"""
    Dưới đây là các câu hỏi trắc nghiệm (có đáp án) về legal multiple choice. Vui lòng chọn đáp án phù hợp nhất cho câu hỏi này.

    Câu hỏi:
    {example['question']}
    {choices_text}
    
    Trả lời: 
    """.strip()
    )

def format_sqa(example): 
    return ( f"""
    Bạn là một chuyên gia pháp lý. 
    Hãy trả lời câu hỏi pháp luật dựa trên kiến thức chuyên môn của mình.
    Khi trả lời:
    - Phân tích pháp lý một cách tự nhiên, như đang nhớ lại và vận dụng kiến thức chuyên môn.
    - Sử dụng các cách diễn đạt như: "Theo quy định tại...", "Căn cứ vào...", "Trong trường hợp này...".
    - Kết thúc bằng một kết luận rõ ràng, trực tiếp trả lời câu hỏi.

    Định dạng đầu ra:
    Phân tích pháp lý: [nội dung phân tích]
    Kết luận: [câu trả lời cụ thể]

    Câu hỏi: {example['question']}
    """.strip()
    )