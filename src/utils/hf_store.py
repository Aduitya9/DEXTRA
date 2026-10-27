"""
Hugging Face dataset-repo storage for DEXTRA data.

Every pipeline stage writes its output here as soon as it is produced and checks
here before doing work, so an interrupted Colab session resumes where it stopped
instead of starting over.
"""

import io
import json

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


class HFStore:
    def __init__(self, repo_id, token, api=None):
        self.repo_id = repo_id
        self.token = token
        self.api = api or HfApi(token=token)
        # Never auto-create: a typo in the repo id must fail loudly, not make a new repo.
        if not self.api.repo_exists(repo_id, repo_type='dataset'):
            raise RuntimeError(
                f"Dataset repo '{repo_id}' was not found or this token cannot see it. "
                "Check HF_DATASET_REPO and that the token has write access to that repo."
            )
        self.refresh()

    def refresh(self):
        self._files = set(self.api.list_repo_files(self.repo_id, repo_type='dataset'))

    def exists(self, path):
        return path in self._files

    def files(self, prefix=''):
        return sorted(f for f in self._files if f.startswith(prefix))

    def _upload(self, fileobj, path, message):
        self.api.upload_file(
            path_or_fileobj=fileobj,
            path_in_repo=path,
            repo_id=self.repo_id,
            repo_type='dataset',
            commit_message=message,
        )
        self._files.add(path)

    def upload_file(self, local_path, path, message):
        self._upload(str(local_path), path, message)

    def upload_df(self, df, path, message):
        """Upload a DataFrame as Parquet (.parquet) or CSV (.csv)."""
        buf = io.BytesIO()
        if path.endswith('.parquet'):
            df.to_parquet(buf, index=False)
        elif path.endswith('.csv'):
            buf.write(df.to_csv(index=False).encode('utf-8'))
        else:
            raise ValueError(f"Unsupported file type for {path}")
        buf.seek(0)
        self._upload(buf, path, message)

    def upload_folder(self, local_dir, path_in_repo, message):
        """One commit for a whole folder (e.g. a batch of CPCB station files)."""
        self.api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type='dataset',
            commit_message=message,
        )
        self.refresh()

    def read_df(self, path):
        local = hf_hub_download(self.repo_id, path, repo_type='dataset', token=self.token)
        return pd.read_parquet(local) if path.endswith('.parquet') else pd.read_csv(local)

    def upload_json(self, obj, path, message):
        self._upload(io.BytesIO(json.dumps(obj, indent=1, sort_keys=True).encode('utf-8')), path, message)

    def read_json(self, path, default=None):
        if not self.exists(path):
            return default
        # force_download: small bookkeeping files change between runs, so never trust a cached copy
        local = hf_hub_download(self.repo_id, path, repo_type='dataset', token=self.token, force_download=True)
        with open(local, encoding='utf-8') as f:
            return json.load(f)
