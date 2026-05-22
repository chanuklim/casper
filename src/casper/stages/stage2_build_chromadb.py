"""Stage 2 — build ChromaDB collections per (split × source).

Two source types are supported:

- ``text``    : raw input from ``data/<split>/*.json`` (field ``text``).
- ``summary`` : keyword-driven summary from
                ``outputs/stage1_keywords_summary/<run>/<split>/*.json`` (field ``response_summ``).

Persist directories are namespaced per source:

    outputs/stage2_chromadb/<run>/<source>/<split>/

Each input file becomes one collection. Collection name encodes the run, source
and split so Stage 3 can identify a paired test/train collection unambiguously.

Speed
-----
Embeddings are computed once per file across both GPUs using
``SentenceTransformer.encode_multi_process``. ChromaDB then receives precomputed
vectors via bulk ``collection.add(...)`` calls, which removes the per-batch
embedding-function lock that bottlenecked the previous implementation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# CUDA_VISIBLE_DEVICES must be set before any torch CUDA call so the
# multi-process pool can see all requested GPUs.
from casper import config

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if config.STAGE2_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.STAGE2_CUDA_VISIBLE_DEVICES

import chromadb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Field used as the embedding text per source.
_SOURCE_FIELDS: Dict[str, str] = {"text": "text", "summary": "response_summ"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_collection_name(name: str) -> str:
    name = name.replace("-", "_").replace(".", "_")
    if len(name) > 63:
        suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        name = f"{name[:54]}_{suffix}"
    return name


def _collection_name(split: str, source: str, file_stem: str) -> str:
    run_part = config.RUN_NAME.replace("-", "_").replace(".", "_")
    return _safe_collection_name(f"{run_part}_{source}_{split}_{file_stem}")


def _resolve_devices() -> List[str]:
    raw = (config.STAGE2_GPU_DEVICES or "").strip()
    if not raw:
        return ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
    return [d.strip() for d in raw.split(",") if d.strip()]


def _resolve_sources() -> List[str]:
    raw = (config.STAGE2_SOURCE or "summary").strip().lower()
    if raw == "both":
        return ["text", "summary"]
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    for p in parts:
        if p not in _SOURCE_FIELDS:
            raise ValueError(f"Unknown stage2 source: {p!r}")
    return parts or ["summary"]


def _load_embed_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model on CPU. Workers in the multi-process pool
    re-instantiate the model on their own target device, so keeping the parent
    on CPU avoids hogging GPU memory in the launcher process.

    ``model_kwargs`` is propagated to the workers as part of the cached
    SentenceTransformer state, so loading in bfloat16 here gives every worker a
    bf16 model on its GPU — ~4x throughput vs. the fp32 default.
    """
    local = snapshot_download(model_name, revision="main", cache_dir=str(config.MODEL_CACHE_DIR))
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(config.STAGE2_DTYPE, torch.bfloat16)
    model = SentenceTransformer(
        local,
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": torch_dtype},
    )
    model.max_seq_length = config.STAGE2_MAX_SEQ_LENGTH
    return model


def _read_records(json_path: Path, text_field: str) -> Tuple[List[str], List[dict], List[str], int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    skipped_empty = 0

    for idx, (key, value) in enumerate(data.items()):
        text = value.get(text_field, "") or ""
        if not text:
            skipped_empty += 1
            continue
        if len(text) > 8000:
            text = text[:8000]

        cn_value = value.get("cn", value.get("CN"))
        meta = {
            "doc_id": str(key),
            "classes_top": json.dumps(value.get("classes_top", []), ensure_ascii=False),
            "classes_sub": json.dumps(value.get("classes_sub", []), ensure_ascii=False),
            "cn": json.dumps(cn_value, ensure_ascii=False) if cn_value is not None else json.dumps([]),
        }
        documents.append(text)
        metadatas.append(meta)
        ids.append(f"doc_{idx}")

    return documents, metadatas, ids, skipped_empty


def _bulk_add(collection, ids, documents, metadatas, embeddings: np.ndarray, chunk: int) -> None:
    n = len(ids)
    for s in tqdm(range(0, n, chunk), desc="  ChromaDB add", unit="chunk"):
        e = min(s + chunk, n)
        collection.add(
            ids=ids[s:e],
            documents=documents[s:e],
            metadatas=metadatas[s:e],
            embeddings=embeddings[s:e].tolist(),
        )


# ---------------------------------------------------------------------------
# Per-file build
# ---------------------------------------------------------------------------
def _build_one(
    *,
    json_path: Path,
    persist_dir: Path,
    split: str,
    source: str,
    model: SentenceTransformer,
    pool,
    overwrite: bool,
) -> dict:
    text_field = _SOURCE_FIELDS[source]
    logger.info(f"\n{'-' * 60}")
    logger.info(f"[{split} / {source}] {json_path.name}")

    documents, metadatas, ids, skipped_empty = _read_records(json_path, text_field)
    total = len(documents)
    if total == 0:
        logger.warning(f"  No records with non-empty '{text_field}' field — skipped")
        return {
            "split": split, "source": source, "source_file": json_path.name,
            "added": 0, "skipped_empty": skipped_empty, "total_in": 0,
        }

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    name = _collection_name(split, source, json_path.stem)
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        if overwrite:
            logger.info(f"  Overwrite mode: deleting existing collection {name}")
            client.delete_collection(name=name)
        else:
            count = client.get_collection(name).count()
            logger.info(
                f"  Append mode: collection {name} already exists with {count:,} docs — skipping. "
                f"Set CASPER_STAGE2_OVERWRITE=true to rebuild."
            )
            return {
                "split": split, "source": source, "source_file": json_path.name,
                "collection_name": name, "added": 0, "skipped_existing": True,
                "total_in": total, "skipped_empty": skipped_empty,
            }

    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    t0 = time.time()
    # New unified API in sentence-transformers 5.x — passing a multi-process pool
    # makes encode() distribute work across the pool's devices.
    embeddings = model.encode(
        documents,
        pool=pool,
        batch_size=config.STAGE2_EMBED_BATCH_SIZE,
        chunk_size=config.STAGE2_ENCODE_CHUNK,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embed_secs = time.time() - t0
    logger.info(
        f"  Embedded {total:,} docs in {embed_secs / 60:.2f} min "
        f"({total / max(1e-6, embed_secs):.1f} docs/s, dim={embeddings.shape[1]})"
    )

    embeddings = embeddings.astype(np.float32, copy=False)

    t1 = time.time()
    _bulk_add(collection, ids, documents, metadatas, embeddings, config.STAGE2_ADD_CHUNK)
    add_secs = time.time() - t1

    final_count = collection.count()
    logger.info(
        f"  Inserted {final_count:,} into {name} "
        f"(add={add_secs:.1f}s, skipped_empty={skipped_empty})"
    )

    return {
        "split": split,
        "source": source,
        "source_file": json_path.name,
        "collection_name": name,
        "added": final_count,
        "skipped_empty": skipped_empty,
        "skipped_existing": False,
        "total_in": total,
        "embed_seconds": embed_secs,
        "add_seconds": add_secs,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    config.ensure_dirs()

    devices = _resolve_devices()
    sources = _resolve_sources()

    logger.info(f"Embedding model : {config.EMBED_MODEL}")
    logger.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"Pool devices    : {devices}")
    logger.info(f"Run name        : {config.RUN_NAME}")
    logger.info(f"Splits          : {list(config.STAGE2_SPLITS)}")
    logger.info(f"Sources         : {sources}")
    logger.info(f"Embed batch     : {config.STAGE2_EMBED_BATCH_SIZE} (max_seq_len={config.STAGE2_MAX_SEQ_LENGTH})")
    logger.info(f"Add chunk       : {config.STAGE2_ADD_CHUNK}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        for d in devices:
            if d.startswith("cuda"):
                idx = int(d.split(":", 1)[1]) if ":" in d else 0
                if idx < torch.cuda.device_count():
                    p = torch.cuda.get_device_properties(idx)
                    logger.info(f"  {d}: {p.name} ({p.total_memory / (1024**3):.1f} GB)")

    model = _load_embed_model(config.EMBED_MODEL)
    logger.info("Starting multi-process pool — workers will load the model on each device...")
    pool = model.start_multi_process_pool(target_devices=devices)

    all_results: List[dict] = []
    started = time.time()
    try:
        for source in sources:
            for split in config.STAGE2_SPLITS:
                in_dir = config.stage2_input_dir(split, source)
                if not in_dir.exists():
                    logger.warning(f"[skip] Input dir missing for split={split} source={source}: {in_dir}")
                    continue

                persist = config.stage2_persist_dir(split, source=source)
                files = sorted(in_dir.glob("*.json"))
                if not files:
                    logger.warning(f"[skip] No JSON files in {in_dir}")
                    continue

                logger.info(f"\n{'=' * 80}")
                logger.info(f"split={split}  source={source}  files={len(files)}  persist={persist}")
                logger.info(f"{'=' * 80}")

                for f in files:
                    res = _build_one(
                        json_path=f,
                        persist_dir=persist,
                        split=split,
                        source=source,
                        model=model,
                        pool=pool,
                        overwrite=config.STAGE2_OVERWRITE_EXISTING,
                    )
                    all_results.append(res)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        try:
            model.stop_multi_process_pool(pool)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to stop multi-process pool cleanly: {exc}")

    elapsed = time.time() - started

    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_name": config.RUN_NAME,
        "embed_model": config.EMBED_MODEL,
        "devices": devices,
        "splits": list(config.STAGE2_SPLITS),
        "sources": sources,
        "total_minutes": elapsed / 60,
        "results": all_results,
    }
    out_root = config.STAGE2_RUN_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    sp = out_root / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nDone in {elapsed / 60:.1f} min — {len(all_results)} files. Summary: {sp}")


if __name__ == "__main__":
    main()
