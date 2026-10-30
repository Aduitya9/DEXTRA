"""
Hugging Face dataset-repo storage for DEXTRA data.

Every pipeline stage writes its output here as soon as it is produced and checks
here before doing work, so an interrupted Colab session resumes where it stopped
instead of starting over.
"""

import json
import tempfile
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

# Worth another try: the request timed out, was rate-limited, or the Hub had a hiccup.
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def is_transient(error):
    """True for a dropped connection or a server-side hiccup; False for a mistake of ours.

    Judged by HTTP status where there is one -- a 401 or a 404 will not change
    on a retry -- and otherwise down the cause chain, because huggingface_hub
    reports a connection dropped mid-download as a FileNotFoundError wrapping
    the network error. Nothing here names an HTTP library: the Hub client has
    changed its own between versions, and Colab installs whichever is current.
    """
    while error is not None:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        if status is not None:
            return status in TRANSIENT_STATUS
        if error.__cause__ is None:
            return not isinstance(error, (TypeError, ValueError, KeyError, AttributeError,
                                          FileNotFoundError, PermissionError, NotImplementedError))
        error = error.__cause__
    return False


class HFStore:
    def __init__(self, repo_id, token, api=None, attempts=4, backoff=5.0):
        self.repo_id = repo_id
        self.token = token
        self.api = api or HfApi(token=token)
        self.attempts, self.backoff = attempts, backoff
        # Never auto-create: a typo in the repo id must fail loudly, not make a new repo.
        if not self._call(self.api.repo_exists, repo_id, repo_type='dataset'):
            raise RuntimeError(
                f"Dataset repo '{repo_id}' was not found or this token cannot see it. "
                "Check HF_DATASET_REPO and that the token has write access to that repo."
            )
        self.refresh()

    def _call(self, fn, *args, **kwargs):
        """Call the Hub, retrying what a short wait can fix and nothing else.

        Retrying a write is safe: a commit that did land before its answer was
        lost comes back as 'no changes', not as a duplicate.
        """
        for attempt in range(1, self.attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as error:
                if attempt == self.attempts or not is_transient(error):
                    raise
                wait = self.backoff * 2 ** (attempt - 1)
                print(f"  Hugging Face: {type(error).__name__}; trying again in {wait:.0f} s "
                      f"(retry {attempt} of {self.attempts - 1})")
                time.sleep(wait)

    def _download(self, path, force=False):
        return self._call(hf_hub_download, self.repo_id, path, repo_type='dataset',
                          token=self.token, force_download=force)

    def refresh(self):
        self._files = set(self._call(self.api.list_repo_files, self.repo_id, repo_type='dataset'))

    def exists(self, path):
        return path in self._files

    def files(self, prefix=''):
        return sorted(f for f in self._files if f.startswith(prefix))

    def _upload(self, fileobj, path, message):
        self._call(
            self.api.upload_file,
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
        """Upload a DataFrame as Parquet (.parquet) or CSV (.csv).

        Written to a temporary file first: Hugging Face's Xet storage only takes
        file paths, and falls back to a slower plain-HTTP upload for in-memory buffers.
        """
        if not path.endswith(('.parquet', '.csv')):
            raise ValueError(f"Unsupported file type for {path}")
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / Path(path).name
            if path.endswith('.parquet'):
                df.to_parquet(local, index=False)
            else:
                df.to_csv(local, index=False)
            self._upload(str(local), path, message)

    def upload_folder(self, local_dir, path_in_repo, message):
        """One commit for a whole folder (e.g. a batch of CPCB station files)."""
        self._call(
            self.api.upload_folder,
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type='dataset',
            commit_message=message,
        )
        self.refresh()

    def read_df(self, path):
        local = self._download(path)
        return pd.read_parquet(local) if path.endswith('.parquet') else pd.read_csv(local)

    def read_bytes(self, path):
        """Raw file contents. force_download: a checkpoint changes between saves."""
        return Path(self._download(path, force=True)).read_bytes()

    def upload_json(self, obj, path, message):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / Path(path).name
            local.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding='utf-8')
            self._upload(str(local), path, message)

    def read_json(self, path, default=None):
        if not self.exists(path):
            return default
        # force_download: small bookkeeping files change between runs, so never trust a cached copy
        local = self._download(path, force=True)
        with open(local, encoding='utf-8') as f:
            return json.load(f)
