#!/usr/bin/env bash
# SE-Bench data (HuggingFace) + API docs and the zwc package (GitHub) -> data/sebench/
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/sebench
HF=https://huggingface.co/datasets/jintailin/SE-Bench/resolve/main
GH=https://raw.githubusercontent.com/thunlp/SE-Bench/main
mkdir -p $D/zwc/zwc/rfx $D/zwc/docs
for f in train single_test multiple_test; do curl -sSL -o $D/$f.jsonl $HF/$f.jsonl; done
curl -sSL -o $D/api_doc.jsonl $GH/datasets/train/api_doc.jsonl
for f in README.md setup.py mappings.json zwc/__init__.py zwc/README.md zwc/mappings.json zwc/rfx/__init__.py \
         docs/function_index.md docs/main_functions.md docs/rfx_functions.md; do
  curl -sSL -o $D/zwc/$f $GH/zwc/$f
done
wc -l $D/*.jsonl
