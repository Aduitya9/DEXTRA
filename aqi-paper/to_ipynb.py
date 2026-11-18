"""Convert pipeline.py (jupytext 'percent' format) into a Colab-ready .ipynb.

The point of this indirection: pipeline.py is directly executable, so the code that
gets tested is byte-for-byte the code that ends up in the notebook cells.

    python to_ipynb.py pipeline.py SIGMANet_AQI.ipynb
"""
import json
import sys
from pathlib import Path


def to_source(text):
    """nbformat wants a list of lines, each keeping its newline except the last."""
    text = text.rstrip("\n")
    if not text:
        return []
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def strip_md(text):
    out = []
    for ln in text.split("\n"):
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.strip() == "#":
            out.append("")
        else:
            out.append(ln)
    return "\n".join(out)


def convert(src_path, out_path):
    raw = Path(src_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    cells = []
    kind, buf = None, []

    def flush():
        nonlocal kind, buf
        if kind is None:
            buf = []
            return
        text = "\n".join(buf).strip("\n")
        if text.strip():
            if kind == "markdown":
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": to_source(strip_md(text))})
            else:
                cells.append({"cell_type": "code", "metadata": {},
                              "execution_count": None, "outputs": [],
                              "source": to_source(text)})
        kind, buf = None, []

    for line in raw.split("\n"):
        s = line.rstrip()
        if s == "# %% [markdown]":
            flush()
            kind = "markdown"
        elif s == "# %%":
            flush()
            kind = "code"
        else:
            if kind is None:          # preamble before the first marker
                kind = "code"
            buf.append(line)
    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "L4", "machine_shape": "hm"},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    Path(out_path).write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"wrote {out_path}: {len(cells)} cells ({n_md} markdown, {len(cells)-n_md} code)")

    # sanity: the notebook must be readable and its code must still compile
    nb2 = json.loads(Path(out_path).read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb2["cells"]
                     if c["cell_type"] == "code")
    compile(code, "<notebook>", "exec")
    print("verified: notebook parses as JSON and all code cells compile together")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "pipeline.py"
    out = sys.argv[2] if len(sys.argv) > 2 else "SIGMANet_AQI.ipynb"
    convert(src, out)
