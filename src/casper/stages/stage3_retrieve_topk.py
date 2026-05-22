"""Stage 3 — for each test/train ChromaDB pair, dump top-K retrieved items to JSON.

Pairs test and train collections by the prefix appearing before ``_test_`` /
``_train_`` in their names. Output rows match the format expected by Stage 4:

    {
      "query_id": ..., "query_text": ..., "classes_top": [...],
      "classes_sub": [...], "cn": [...],
      "retrieved": [
        {"id": ..., "classes_top": ..., "classes_sub": ..., "cn": ...,
         "distance": ..., "document": ...}, ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from tqdm import tqdm

from casper import config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TopKExporter")


def _safe_to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return list(x)
    except Exception:
        return [x]


def _parse_meta_field(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _jsonl_to_json(jsonl_path: Path, json_path: Path) -> None:
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(json_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if not first:
                fout.write(",\n")
            fout.write(line)
            first = False
        fout.write("\n]\n")


def _format_retrieved(r_ids, r_metas, r_dists, r_docs) -> List[dict]:
    out: List[dict] = []
    m = min(len(r_ids), len(r_metas), len(r_dists), len(r_docs))
    for j in range(m):
        rmeta = r_metas[j] or {}
        out.append({
            "id": r_ids[j],
            "classes_top": _parse_meta_field(rmeta.get("classes_top")),
            "classes_sub": _parse_meta_field(rmeta.get("classes_sub")),
            "cn": _parse_meta_field(rmeta.get("cn")),
            "distance": float(r_dists[j]) if r_dists else None,
            "document": r_docs[j],
        })
    return out


def _format_row(qid: str, qmeta: dict, qdoc, retrieved: List[dict]) -> dict:
    return {
        "query_id": qid,
        "doc_id": qmeta.get("doc_id"),
        "query_text": qdoc,
        "classes_top": _parse_meta_field(qmeta.get("classes_top")),
        "classes_sub": _parse_meta_field(qmeta.get("classes_sub")),
        "cn": _parse_meta_field(qmeta.get("cn")),
        "retrieved": retrieved,
    }


def _prefix_before_test_train(name: str) -> Optional[str]:
    if "_test_" in name:
        return name.split("_test_", 1)[0]
    if "_train_" in name:
        return name.split("_train_", 1)[0]
    return None


class TopKExporter:
    def __init__(
        self,
        train_persist_dir: Path,
        test_persist_dir: Path,
        out_dir: Path,
        k: int = config.STAGE3_TOP_K,
        batch_size: int = config.STAGE3_QUERY_BATCH_SIZE,
        tqdm_disable: bool = False,
    ):
        self.train_client = chromadb.PersistentClient(
            path=str(train_persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self.test_client = chromadb.PersistentClient(
            path=str(test_persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self.k = int(k)
        self.batch_size = int(batch_size)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tqdm_disable = tqdm_disable
        logger.info(f"Initialized TopKExporter | k={self.k}, batch_size={self.batch_size}, out_dir={self.out_dir}")

    def find_target_pairs(self) -> List[Tuple[str, str]]:
        """Pair `<prefix>_test_*` with `<prefix>_train_*` collections."""
        test_names = sorted(c.name for c in self.test_client.list_collections())
        train_names = sorted(c.name for c in self.train_client.list_collections())

        logger.info(f"TEST collections ({len(test_names)}): {test_names}")
        logger.info(f"TRAIN collections ({len(train_names)}): {train_names}")

        train_by_prefix: Dict[str, List[str]] = defaultdict(list)
        for tr in train_names:
            p = _prefix_before_test_train(tr)
            if p is not None and "_train_" in tr:
                train_by_prefix[p].append(tr)

        pairs: List[Tuple[str, str]] = []
        unmatched: List[str] = []

        for t in test_names:
            if "_test_" not in t:
                continue
            prefix = _prefix_before_test_train(t)
            if prefix is None:
                unmatched.append(t)
                continue

            expected = t.replace("_test_", "_train_", 1)
            if expected in train_names:
                pairs.append((t, expected))
                continue

            candidates = train_by_prefix.get(prefix, [])
            if len(candidates) == 1:
                pairs.append((t, candidates[0]))
            elif len(candidates) > 1:
                def _score(name: str) -> int:
                    a = set(expected.split("_"))
                    b = set(name.split("_"))
                    return len(a & b)

                best = sorted(candidates, key=_score, reverse=True)[0]
                pairs.append((t, best))
                logger.warning(
                    f"Multiple train candidates for prefix='{prefix}'. "
                    f"Picked best='{best}' from {candidates}"
                )
            else:
                unmatched.append(t)

        logger.info(f"Target pairs: {len(pairs)}")
        if unmatched:
            logger.warning(f"Unmatched TEST collections: {unmatched}")

        return sorted(pairs, key=lambda x: (x[0], x[1]))

    def _fetch_test_batch(self, test_col, offset: int, limit: int):
        docs = test_col.get(offset=offset, limit=limit, include=["embeddings", "metadatas", "documents"])
        return {
            "ids": _safe_to_list(docs.get("ids")),
            "embeddings": _safe_to_list(docs.get("embeddings")),
            "metadatas": _safe_to_list(docs.get("metadatas")),
            "documents": _safe_to_list(docs.get("documents")),
        }

    def export_pair(
        self,
        test_name: str,
        train_name: str,
        limit_queries: Optional[int] = None,
        write_json_array: bool = True,
    ) -> Optional[Path]:
        try:
            test_col = self.test_client.get_collection(name=test_name)
            train_col = self.train_client.get_collection(name=train_name)
        except Exception as e:
            logger.error(f"Failed to open collections {test_name} / {train_name}: {e}")
            return None

        total = int(test_col.count())
        if limit_queries is not None:
            total = min(total, int(limit_queries))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{test_name}__{train_name}__top{self.k}_{ts}"
        out_jsonl = self.out_dir / f"{base}.jsonl"
        out_json = self.out_dir / f"{base}.json"

        logger.info(f"\nExporting: {test_name} -> {train_name} (top-{self.k}, queries={total})")

        written = 0
        with open(out_jsonl, "w", encoding="utf-8") as fout, tqdm(
            total=total, desc=f"Processing {test_name}", disable=self.tqdm_disable
        ) as pbar:
            processed = 0
            while processed < total:
                fetch = min(self.batch_size, total - processed)
                batch = self._fetch_test_batch(test_col, offset=processed, limit=fetch)
                ids = batch["ids"]
                embs = batch["embeddings"]
                metas = batch["metadatas"]
                docs = batch["documents"]

                n = min(len(ids), len(embs), len(metas), len(docs))
                if n == 0:
                    continue

                # Batched query — chromadb processes the whole batch internally and
                # returns parallel lists keyed by query position.
                try:
                    batch_res = train_col.query(
                        query_embeddings=[embs[i] for i in range(n)],
                        n_results=self.k,
                        include=["metadatas", "distances", "documents"],
                    )
                except Exception as e:
                    logger.warning(f"Batched query failed for {test_name} ({n} queries): {e}")
                    batch_res = None

                if batch_res is None:
                    # Fall back per-query so a single bad row doesn't kill the file.
                    for i in range(n):
                        qid = ids[i]
                        try:
                            res = train_col.query(
                                query_embeddings=[embs[i]],
                                n_results=self.k,
                                include=["metadatas", "distances", "documents"],
                            )
                        except Exception as e:
                            logger.warning(f"Per-query fallback failed for {qid}: {e}")
                            continue
                        r_ids = _safe_to_list(res.get("ids", [[]])[0])
                        r_metas = _safe_to_list(res.get("metadatas", [[]])[0])
                        r_dists = _safe_to_list(res.get("distances", [[]])[0])
                        r_docs = _safe_to_list(res.get("documents", [[]])[0])
                        retrieved = _format_retrieved(r_ids, r_metas, r_dists, r_docs)
                        row = _format_row(ids[i], metas[i] or {}, docs[i] if i < len(docs) else None, retrieved)
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                        processed += 1
                        pbar.update(1)
                    continue

                # Successful batched response — slice per query.
                all_r_ids = _safe_to_list(batch_res.get("ids", []))
                all_r_metas = _safe_to_list(batch_res.get("metadatas", []))
                all_r_dists = _safe_to_list(batch_res.get("distances", []))
                all_r_docs = _safe_to_list(batch_res.get("documents", []))

                for i in range(n):
                    r_ids = _safe_to_list(all_r_ids[i]) if i < len(all_r_ids) else []
                    r_metas = _safe_to_list(all_r_metas[i]) if i < len(all_r_metas) else []
                    r_dists = _safe_to_list(all_r_dists[i]) if i < len(all_r_dists) else []
                    r_docs = _safe_to_list(all_r_docs[i]) if i < len(all_r_docs) else []
                    retrieved = _format_retrieved(r_ids, r_metas, r_dists, r_docs)
                    row = _format_row(ids[i], metas[i] or {}, docs[i] if i < len(docs) else None, retrieved)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    processed += 1
                    pbar.update(1)

        if written == 0:
            logger.warning("No rows were written.")
            return None

        logger.info(f"Wrote JSONL: {out_jsonl} (rows={written})")

        if write_json_array:
            _jsonl_to_json(out_jsonl, out_json)
            logger.info(f"Wrote JSON array: {out_json}")

        return out_json if write_json_array else out_jsonl


def main() -> None:
    config.ensure_dirs()

    source = config.STAGE3_SOURCE
    train_dir = config.stage2_persist_dir("train", source=source)
    test_dir = config.stage2_persist_dir("test", source=source)
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit(
            f"ChromaDB dirs missing for source={source!r} — "
            f"train: {train_dir} (exists={train_dir.exists()}), "
            f"test: {test_dir} (exists={test_dir.exists()}). Run Stage 2 first."
        )

    out_dir = config.stage3_source_output_dir()
    logger.info(f"Stage 3 source = {source}  out_dir = {out_dir}")
    exporter = TopKExporter(
        train_persist_dir=train_dir,
        test_persist_dir=test_dir,
        out_dir=out_dir,
    )

    pairs = exporter.find_target_pairs()
    if not pairs:
        raise SystemExit("No matching test/train collection pairs found.")

    logger.info(f"Discovered pairs: {pairs}")
    for test_name, train_name in pairs:
        exporter.export_pair(test_name, train_name)


if __name__ == "__main__":
    main()
