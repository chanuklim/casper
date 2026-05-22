#!/usr/bin/env bash
# Stage 1 — keyword extraction + category-driven summarization.
# Override the split via:  CASPER_STAGE1_SPLIT=train ./scripts/run_stage1.sh
source "$(dirname "$0")/_common.sh"
exec python -m casper.stages.stage1_keywords_summary "$@"
