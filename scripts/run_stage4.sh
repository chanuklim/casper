#!/usr/bin/env bash
# Stage 4 — classify with retrieved examples for every (class_type, k) pair.
source "$(dirname "$0")/_common.sh"
exec python -m casper.stages.stage4_classify "$@"
