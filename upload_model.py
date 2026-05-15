import argparse
from pathlib import Path
from huggingface_hub import login, create_repo, upload_folder
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

parser = argparse.ArgumentParser('Interface for uploading merged model to Huggingface!')
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

def upload_model(args, username):
    model_path = Path(args.model_path)
    model_path = Path(*model_path.parts[1:])
    parts = model_path.parts

    repo_name = '-'.join([
        parts[-6], parts[-5], parts[-4], f'epoch{parts[-3]}', f'lr{parts[-2]}', f'val{parts[-1]}'
    ])

    repo_id = f'{username}/{repo_name}'
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(HF_TOKEN)
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