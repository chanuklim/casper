"""Project-wide configuration: paths, model names, hyperparameters.

All stages import from this module so paths stay consistent end-to-end.
Override any constant by setting the matching environment variable
(prefix `CASPER_`) before launching a stage.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"CASPER_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# src/casper/config.py -> parents[2] is project root (casper/)

DATA_ROOT = Path(_env("DATA_ROOT", str(PROJECT_ROOT / "data")))
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
STAGE1_DIR = OUTPUTS_ROOT / "stage1_keywords_summary"
STAGE2_DIR = OUTPUTS_ROOT / "stage2_chromadb"
STAGE3_DIR = OUTPUTS_ROOT / "stage3_retrieved"
STAGE4_DIR = OUTPUTS_ROOT / "stage4_classification"

MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Generative LLM used in Stage 1 (keywords + summary) and Stage 4 (classify).
GEN_MODEL = _env("GEN_MODEL", "gsjang/lim-4b-1-0826")
GEN_MODEL_SHORT = GEN_MODEL.split("/")[-1]

# Embedding model used in Stage 2 (vectorize summaries).
EMBED_MODEL = _env("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")

# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
RATIO = _env_float("RATIO", 0.2)
RUN_NAME = _env("RUN_NAME", f"{GEN_MODEL_SHORT}_ratio_{RATIO}")

# Per-run subdir helpers
STAGE1_RUN_DIR = STAGE1_DIR / RUN_NAME
STAGE2_RUN_DIR = STAGE2_DIR / RUN_NAME  # ChromaDB persist roots: STAGE2_RUN_DIR/{train,test}
STAGE3_RUN_DIR = STAGE3_DIR / RUN_NAME
STAGE4_RUN_DIR = STAGE4_DIR / RUN_NAME

# ---------------------------------------------------------------------------
# GPU / vLLM
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES = _env("CUDA_VISIBLE_DEVICES", "0")
GPU_ID = _env_int("GPU_ID", 0)

VLLM_GPU_MEMORY_UTILIZATION_STAGE1 = _env_float("VLLM_GPU_UTIL_STAGE1", 0.90)
VLLM_GPU_MEMORY_UTILIZATION_STAGE4 = _env_float("VLLM_GPU_UTIL_STAGE4", 0.95)
VLLM_MAX_MODEL_LEN_STAGE1 = _env_int("VLLM_MAX_LEN_STAGE1", 8192)
VLLM_MAX_MODEL_LEN_STAGE4 = _env_int("VLLM_MAX_LEN_STAGE4", 30000)
VLLM_MAX_NUM_SEQS = _env_int("VLLM_MAX_NUM_SEQS", 512)
VLLM_MAX_NUM_BATCHED_TOKENS = _env_int("VLLM_MAX_NUM_BATCHED_TOKENS", 32768)
# Tensor-parallel size for vLLM. Set to 2 to shard a single model across both A100s.
VLLM_TENSOR_PARALLEL_SIZE = _env_int("VLLM_TP", 1)

# ---------------------------------------------------------------------------
# Stage 1 (keywords + summary)
# ---------------------------------------------------------------------------
STAGE1_BATCH_SIZE = _env_int("STAGE1_BATCH_SIZE", 256)
STAGE1_SUB_BATCH_SIZE = _env_int("STAGE1_SUB_BATCH_SIZE", 128)
STAGE1_MAX_RETRIES = _env_int("STAGE1_MAX_RETRIES", 5)
STAGE1_RETRY_DELAY = _env_float("STAGE1_RETRY_DELAY", 1.0)
STAGE1_KEYWORD_MIN_NONEMPTY = _env_int("STAGE1_KW_MIN_NONEMPTY", 5)

# Default split processed by Stage 1 — override with CASPER_STAGE1_SPLIT=train.
STAGE1_SPLIT = _env("STAGE1_SPLIT", "test")

# ---------------------------------------------------------------------------
# Stage 2 (ChromaDB)
# ---------------------------------------------------------------------------
STAGE2_STORAGE_BATCH_SIZE = _env_int("STAGE2_STORAGE_BATCH_SIZE", 75)
STAGE2_EMBED_BATCH_SIZE = _env_int("STAGE2_EMBED_BATCH_SIZE", 256)
STAGE2_MAX_SEQ_LENGTH = _env_int("STAGE2_MAX_SEQ_LENGTH", 512)
STAGE2_NUM_WORKERS = _env_int("STAGE2_NUM_WORKERS", 8)
STAGE2_OVERWRITE_EXISTING = _env("STAGE2_OVERWRITE", "false").lower() in {"1", "true", "yes"}
STAGE2_SPLITS = tuple(s for s in _env("STAGE2_SPLITS", "train,test").split(",") if s)
# Which input goes into the index. "text" embeds raw `text` from data/<split>/; "summary"
# embeds Stage-1 `response_summ`; "both" runs both passes back-to-back.
STAGE2_SOURCE = _env("STAGE2_SOURCE", "both")
# Devices used by the sentence-transformers multi-process pool. e.g. "cuda:0,cuda:1".
STAGE2_GPU_DEVICES = _env("STAGE2_GPU_DEVICES", "cuda:0,cuda:1")
# CUDA_VISIBLE_DEVICES applied at stage-2 startup (independent of CASPER_CUDA_VISIBLE_DEVICES).
STAGE2_CUDA_VISIBLE_DEVICES = _env("STAGE2_CUDA_VISIBLE_DEVICES", "0,1")
# Per-call chunk size when bulk-inserting into ChromaDB (must be < client.get_max_batch_size()=5461).
STAGE2_ADD_CHUNK = _env_int("STAGE2_ADD_CHUNK", 2000)
# Round-robin chunk size for encode_multi_process (texts shipped to each worker per round).
STAGE2_ENCODE_CHUNK = _env_int("STAGE2_ENCODE_CHUNK", 2048)
# torch dtype for the embedding model. bfloat16 ≈ 4x throughput vs fp32 on A100,
# numerically safe for cosine similarity. Override with "float32" if needed.
STAGE2_DTYPE = _env("STAGE2_DTYPE", "bfloat16")

# ---------------------------------------------------------------------------
# Stage 3 (retrieve top-K)
# ---------------------------------------------------------------------------
STAGE3_TOP_K = _env_int("STAGE3_TOP_K", 50)
STAGE3_QUERY_BATCH_SIZE = _env_int("STAGE3_QUERY_BATCH_SIZE", 512)

# ---------------------------------------------------------------------------
# Stage 4 (classify)
# ---------------------------------------------------------------------------
STAGE4_TOP_K = tuple(int(x) for x in _env("STAGE4_TOP_K", "5,10,15,20,25,30,40,50").split(","))
STAGE4_CLASS_TYPES = tuple(s for s in _env("STAGE4_CLASS_TYPES", "sub,top").split(",") if s)
STAGE4_RANDOM_SEED = _env_int("STAGE4_RANDOM_SEED", 12345)
STAGE4_MAX_DATA_NUM = _env_int("STAGE4_MAX_DATA_NUM", 5000)
STAGE4_MAX_RETRIES = _env_int("STAGE4_MAX_RETRIES", 5)
STAGE4_RETRY_DELAY = _env_float("STAGE4_RETRY_DELAY", 1.0)
STAGE4_SAVE_INTERVAL = _env_int("STAGE4_SAVE_INTERVAL", 10)
# Batched-generation chunk size — number of prompts handed to vLLM per `model.generate()` call.
STAGE4_BATCH_SIZE = _env_int("STAGE4_BATCH_SIZE", 128)


def stage1_input_dir(split: str) -> Path:
    """Raw dataset directory for a split."""
    return DATA_ROOT / split


def stage1_output_dir(split: str) -> Path:
    """Stage 1 keyword+summary output directory for a split."""
    return STAGE1_RUN_DIR / split


def stage2_persist_dir(split: str, source: str = "summary") -> Path:
    """ChromaDB persist directory for a (source, split) pair.

    Default ``source="summary"`` keeps backward compatibility with Stage 3,
    which currently hard-codes that branch.
    """
    return STAGE2_RUN_DIR / source / split


def stage2_input_dir(split: str, source: str) -> Path:
    """Where Stage 2 reads its records from for a given (source, split)."""
    if source == "text":
        return DATA_ROOT / split
    if source == "summary":
        return stage1_output_dir(split)
    raise ValueError(f"Unknown stage2 source: {source!r}")


def stage3_output_dir() -> Path:
    return STAGE3_RUN_DIR


def stage4_output_dir() -> Path:
    return STAGE4_RUN_DIR


# ---------------------------------------------------------------------------
# Stage-1 toggle: when true, only keyword extraction is done, summary is left alone.
# ---------------------------------------------------------------------------
STAGE1_SUMMARY_SKIP = _env("STAGE1_SUMMARY_SKIP", "false").lower() in {"1", "true", "yes"}

# Stage-3 source-aware path. `summary` keeps the legacy dir; others go in a subdir.
STAGE3_SOURCE = _env("STAGE3_SOURCE", "summary")


def stage3_source_output_dir() -> Path:
    """Stage-3 output dir; legacy `summary` stays at run-dir, others get a subdir."""
    if STAGE3_SOURCE == "summary":
        return STAGE3_RUN_DIR
    return STAGE3_RUN_DIR / STAGE3_SOURCE


def ensure_dirs() -> None:
    for d in (
        OUTPUTS_ROOT,
        STAGE1_DIR,
        STAGE2_DIR,
        STAGE3_DIR,
        STAGE4_DIR,
        MODEL_CACHE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# Point every HuggingFace-aware library (transformers, sentence-transformers,
# huggingface_hub, vLLM tokenizer downloads) at MODEL_CACHE_DIR. Set unless the
# user already configured HF caching themselves.
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR))
