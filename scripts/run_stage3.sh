#!/usr/bin/env bash
# Stage 3 — export top-K retrieved items from each test/train collection pair.
source "$(dirname "$0")/_common.sh"
exec python -m casper.stages.stage3_retrieve_topk "$@"
