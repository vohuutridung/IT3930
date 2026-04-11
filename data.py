from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from torch.utils.data import DataLoader
import torch


config = {
    'model_id': 'vohuutridung/qwen3-1.7b-legal-pretrain',
    'max_length': 1024,
    'shot_per_task': 8,
    'tasks': [
        {
            'name': 'nli',
            'dataset_id': 'vohuutridung/train-nli',
        },
        {
            'name': 'mcq',
            'dataset_id': 'vohuutridung/vlsp-mcq-v2',
        },
        {
            'name': 'sqa',
            'dataset_id': 'vohuutridung/train-sqa',
        },
    ]
}


class FewShotPipeline:
    """
    Returns: Dataset with (shot_per_task * len(tasks)) with features ('task_id', 'input_ids', 'attention_mask'), 
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'])

    def load_ds(self, task):
        ds = load_dataset(task['dataset_id'], split='train')
        ds = ds.select(range(self.config['shot_per_task']))
        return ds

    def format_ds(self, ds, task, task_id):
        format_fn = FORMAT_REGISTRY[task['name']]

        def map_fn(example):
            return {
                'text': format_fn(example),
                'task_id': task_id
            }
        
        ds = ds.map(map_fn)

        keep_cols = ['text', 'task_id']
        ds = ds.remove_columns([col for col in ds.column_names if col not in keep_cols])

        return ds

    def tokenize_ds(self, ds):
        def tokenize_fn(example):
            return self.tokenizer(
                example['text'],
                padding='max_length',
                truncation=True,
                max_length=self.config['max_length'],
            )
        
        return ds.map(tokenize_fn)
    
    def build(self):
        all_ds = []

        for i, task in enumerate(self.config['tasks']):
            task_name = task['name']  # avoid nested quotes inside f-string (Python < 3.12)
            print(f'loading dataset {task_name}...')
            ds = self.load_ds(task)

            print(f'formatting dataset {task_name}...')
            ds = self.format_ds(ds, task, i)

            print(f'tokenizing dataset {task_name}...')
            ds = self.tokenize_ds(ds)

            all_ds.append(ds)
        
        return all_ds  # list[Dataset] — one Dataset per task, columns: ['task_id', 'input_ids', 'attention_mask']

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

FORMAT_REGISTRY = {
    'nli': format_nli,
    'mcq': format_mcq,
    'sqa': format_sqa,
}


import random
import torch
from torch.utils.data import DataLoader


class MultiTaskDataLoader:
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size

        self.loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in datasets
        ]

        # số batch mỗi task
        self.lengths = [len(loader) for loader in self.loaders]

        # tổng số batch
        self.total_batches = sum(self.lengths)

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        # tạo iterator mới mỗi epoch
        self.iters = [iter(loader) for loader in self.loaders]

        # tạo schedule các task (shuffle ở level batch)
        self.task_schedule = []
        for task_id, num_batches in enumerate(self.lengths):
            self.task_schedule.extend([task_id] * num_batches)

        random.shuffle(self.task_schedule)

        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.total_batches:
            raise StopIteration

        task_id = self.task_schedule[self.idx]
        self.idx += 1

        batch = next(self.iters[task_id])

        return {
            "data": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "source_loader": torch.tensor(task_id),
        }