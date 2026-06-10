import argparse
from pathlib import Path
from huggingface_hub import login, create_repo, upload_folder
import os
from dotenv import load_dotenv

load_dotenv() 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

parser = argparse.ArgumentParser('Interface for uploading merged model to Huggingface!')
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

def upload_model(args, username):
    import re
    # Resolve the model path that exists on disk
    model_path = Path(args.model_path)
    if not model_path.exists():
        parts = model_path.parts
        for i in range(1, len(parts)):
            candidate = Path(*parts[i:])
            if candidate.exists():
                model_path = candidate
                break

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model path: {args.model_path}")

    parts = model_path.parts

    repo_name = '-'.join([
        parts[-6], parts[-5], parts[-4], f'epoch{parts[-3]}', f'lr{parts[-2]}', f'val{parts[-1]}'
    ])

    # Try to load granularity from train_*.log in the model path
    granularity = 'elementwise'
    val_shot = parts[-1]
    log_path = model_path / f'train_{val_shot}.log'
    if not log_path.exists():
        log_files = list(model_path.glob('train_*.log'))
        if log_files:
            log_path = log_files[0]

    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if 'Configuration is' in line:
                        match = re.search(r"granularity=['\"]([^'\"]+)['\"]", line)
                        if match:
                            granularity = match.group(1)
                            break
        except Exception as e:
            print(f"Warning: Failed to parse log file {log_path}: {e}")

    if granularity != 'elementwise':
        repo_name += f'-{granularity}'

    repo_id = f'{username}/{repo_name}'
    login("")
    create_repo(repo_id, repo_type="model", exist_ok=True)

    readme_path = model_path / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f'# {repo_name}')

    upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type='model',
    )

    print("Uploaded:", repo_id)


if __name__ == '__main__':
    upload_model(args, 'vohuutridung')