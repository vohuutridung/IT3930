import numpy as np
import pandas as pd
import os
import subprocess
import sys
from dotenv import load_dotenv
import argparse
import tarfile
import json
import shutil
from pathlib import Path


load_dotenv()

parser = argparse.ArgumentParser('Interface for hmodel_to_kdataset')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--no-cleanup', action='store_true',
                    help='Keep the temporary kdataset directory after upload')
args = parser.parse_args()


def upload_to_kaggle(args, username):
    model_path = Path(args.model_path)
    model_path = Path(*model_path.parts[1:])
    parts = model_path.parts

    repo_name = '-'.join([
        parts[-6], parts[-5], parts[-4], f'epoch{parts[-3]}', f'lr{parts[-2]}', f'val{parts[-1]}'
    ])

    dataset_slug = repo_name.replace('.', '-').lower()
    repo_id = f'{username}/{dataset_slug}'

    # ── 1. Pack model directory into a tar.gz archive ──────────────────────────
    TAR_PATH = "./my_model.tar.gz"
    print(f"Packing model from: {model_path}")
    with tarfile.open(TAR_PATH, "w:gz") as tar:
        for filename in os.listdir(model_path):
            filepath = os.path.join(model_path, filename)
            tar.add(filepath, arcname=filename)
    print("Packed to:", TAR_PATH)

    # ── 2. Prepare the Kaggle dataset directory ─────────────────────────────────
    DATASET_DIR = "./kdataset"
    os.makedirs(DATASET_DIR, exist_ok=True)

    shutil.move(TAR_PATH, os.path.join(DATASET_DIR, "my_model.tar.gz"))

    metadata = {
        "title": repo_name,
        "id": repo_id,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": False,
    }

    with open(os.path.join(DATASET_DIR, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Dataset metadata written.")

    # ── 3. Configure Kaggle credentials from environment ───────────────────────
    kaggle_username = os.getenv("KAGGLE_USERNAME", username)
    kaggle_key = os.getenv("KAGGLE_KEY")
    if not kaggle_key:
        raise EnvironmentError(
            "KAGGLE_KEY is not set. Add it to your .env file or environment variables."
        )
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    # ── 4. Upload: create dataset or add a new version if it already exists ────
    print(f"Uploading dataset to Kaggle as: {repo_id}")

    create_result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", DATASET_DIR, "--dir-mode", "zip"],
        capture_output=True,
        text=True,
    )

    if create_result.returncode == 0:
        print("Dataset created successfully!")
        print(create_result.stdout)
    else:
        # Dataset likely already exists — push a new version instead
        stderr_lower = create_result.stderr.lower() + create_result.stdout.lower()
        if "already in use" in stderr_lower or "already exists" in stderr_lower or "conflict" in stderr_lower:
            print("Dataset already exists. Creating a new version...")
            version_note = f"Auto-update: {repo_name}"
            update_result = subprocess.run(
                [
                    "kaggle", "datasets", "version",
                    "-p", DATASET_DIR,
                    "-m", version_note,
                    "--dir-mode", "zip",
                ],
                capture_output=True,
                text=True,
            )
            if update_result.returncode == 0:
                print("New dataset version created successfully!")
                print(update_result.stdout)
            else:
                print("Error creating new dataset version:", file=sys.stderr)
                print(update_result.stderr, file=sys.stderr)
                sys.exit(update_result.returncode)
        else:
            print("Error creating dataset:", file=sys.stderr)
            print(create_result.stderr, file=sys.stderr)
            sys.exit(create_result.returncode)

    # ── 5. Cleanup ──────────────────────────────────────────────────────────────
    if not args.no_cleanup:
        shutil.rmtree(DATASET_DIR)
        print("Cleaned up temporary dataset directory.")

    print(f"Done! Dataset available at: https://www.kaggle.com/datasets/{repo_id}")


if __name__ == '__main__':
    upload_to_kaggle(args, username='vodung020905')

