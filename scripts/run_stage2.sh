#!/usr/bin/env bash
# Stage 2 — embed input docs into ChromaDB (per split, per source type).
# Defaults:
#   CASPER_STAGE2_SPLITS=train,test
#   CASPER_STAGE2_SOURCE=both           # "text" | "summary" | "both"
#   CASPER_STAGE2_GPU_DEVICES=cuda:0,cuda:1
#   CASPER_STAGE2_CUDA_VISIBLE_DEVICES=0,1
# Override any of these in the env before invoking the script.
source "$(dirname "$0")/_common.sh"

export CASPER_STAGE2_CUDA_VISIBLE_DEVICES="${CASPER_STAGE2_CUDA_VISIBLE_DEVICES:-0,1}"
export CASPER_STAGE2_GPU_DEVICES="${CASPER_STAGE2_GPU_DEVICES:-cuda:0,cuda:1}"
export CASPER_STAGE2_SOURCE="${CASPER_STAGE2_SOURCE:-both}"

exec python -m casper.stages.stage2_build_chromadb "$@"
