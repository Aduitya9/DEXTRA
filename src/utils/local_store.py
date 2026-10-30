"""
A local-filesystem stand-in for HFStore, with the same small interface.

CPCB blocks the cloud IP ranges Colab runs on: every request from a notebook
times out at connect, while the same URL answers in under a second from an
ordinary Indian connection. So the station files have to be fetched on a normal
machine, and this lets the same, tested `download_cpcb.download_manifest` write
to a folder instead of to Hugging Face -- no second code path to keep honest.
"""

import json
import shutil
from pathlib import Path

import pandas as pd


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def refresh(self):
        pass

    def _relative(self, path):
        return str(Path(path).relative_to(self.root)).replace('\\', '/')

    def exists(self, path):
        return (self.root / path).is_file()

    def files(self, prefix=''):
        return sorted(self._relative(p) for p in self.root.rglob('*')
                      if p.is_file() and self._relative(p).startswith(prefix))

    def upload_folder(self, local_dir, path_in_repo, message):
        """Move a batch into place. Named for the HFStore method it stands in for."""
        target = self.root / path_in_repo
        target.mkdir(parents=True, exist_ok=True)
        for item in Path(local_dir).iterdir():
            shutil.copy2(item, target / item.name)

    def upload_file(self, local_path, path, message):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)

    def read_bytes(self, path):
        return (self.root / path).read_bytes()

    def upload_json(self, obj, path, message):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding='utf-8')

    def read_json(self, path, default=None):
        target = self.root / path
        return json.loads(target.read_text(encoding='utf-8')) if target.is_file() else default

    def upload_df(self, df, path, message):
        """Parquet or CSV by extension, as HFStore.upload_df does."""
        if not path.endswith(('.parquet', '.csv')):
            raise ValueError(f"Unsupported file type for {path}")
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith('.parquet'):
            df.to_parquet(target, index=False)
        else:
            df.to_csv(target, index=False)

    def read_df(self, path):
        target = self.root / path
        return pd.read_parquet(target) if path.endswith('.parquet') else pd.read_csv(target)
