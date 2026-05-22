# CASPER

Reference implementation of **CASPER** — *a Classification System using Semantically
Compressed Retrieval-Augmentation*.

CASPER targets production-scale classification of scientific and technical
documents (scholarly papers, patents, R&D reports) under fine-grained
taxonomies with hundreds to thousands of labels. It inserts an
**entity-centric semantic compressor** between retrieval and inference: each
document is distilled into a structured set of salient entities (e.g.
*core concepts*, *methodologies*, *findings*) and rewritten into a compact,
task-focused representation. Both the vector index and the per-query prompt
are built from these compressed inputs, so retrieval-augmented in-context
classification runs on short, label-discriminative text instead of raw long
documents.

### Why compression matters here

Retrieval-augmented in-context classification scales naturally to evolving
taxonomies and large label spaces, but reasoning over many long retrieved
exemplars inflates inference cost and degrades accuracy when critical
evidence sits mid-context. Length truncation, TF-IDF keyword selection,
sentence ranking, free-form LLM summarization, and prior learned compressors
for RAG (LLMLingua, RECOMP) all either match or hurt accuracy versus raw
retrieved text on this task. The structured, entity-centric selection step is
what isolates the gain: holding the trained 4 B classifier fixed and only
swapping its input from raw retrieval to CASPER-compressed input lifts
domain-averaged fine-grained classification by **+3.4 / +5.2 pp Micro- /
Macro-F1**, and a counterfactual same-type-entity swap confirms the effect is
causal — the entities CASPER selects carry the label-discriminative signal,
the discarded ones are approximately redundant. At a 20 % compression ratio
the input length is roughly halved (≈ 50 % token reduction averaged across
domains) within noise of full-input accuracy, and the compressed index
preserves both the coverage and ranking of label-relevant evidence.

## Pipeline at a glance

CASPER is a two-phase pipeline:

- **Phase 1 (index build).** For every labeled document in the corpus, the
  compressor extracts entities of seven predefined types, selects the
  top-`m` most informative ones (compression ratio `r = m / M`), and rewrites
  them into a compact, structured summary `d̃`. Summaries are embedded with a
  sentence-embedding model and stored in a vector database.
- **Phase 2 (classify a query).** A new document is run through the same
  compressor; top-K nearest items are retrieved from the vector index;
  the LLM classifies the query conditioned on the retrieved compressed
  exemplars.

The seven entity types are
`core_concepts`, `methodologies`, `subjects_problems`, `findings_impacts`,
`theoretical_framework`, `quantitative_metrics`, `contextual_background` —
ordered roughly by classification-discriminative weight, with
`core_concepts` the most label-bearing and `contextual_background` the most
removable. They are listed in the Stage 1 prompts in `src/casper/prompts.py`.

Code layout follows the same logical decomposition:

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `stage1_keywords_summary.py` | `data/<split>/*.json` | `outputs/stage1_keywords_summary/<run>/<split>/*.json` (keywords + compressed summary) |
| 2 | `stage2_build_chromadb.py`   | Stage 1 outputs (or raw text)  | `outputs/stage2_chromadb/<run>/<source>/<split>/` (ChromaDB persist dir) |
| 3 | `stage3_retrieve_topk.py`    | Stage 2 ChromaDB collections   | `outputs/stage3_retrieved/<run>/*.json` (per-query top-K) |
| 4 | `stage4_classify.py`         | Stage 3 retrieved JSON         | `outputs/stage4_classification/<run>/<class_type>/k_<K>/*.json` |

Stage 2 supports two index sources: `text` (raw retrieved documents — the
no-compression baseline) and `summary` (CASPER-compressed text, the default
index). Stage 1 and Stage 4 are resumable per output file — re-running on
an existing output skips items that already have a saved response.

## Quickstart (sample data)

```bash
git clone <this-repo>.git casper && cd casper
pip install -r requirements.txt

# Point CASPER at the bundled sample dataset (5 docs per source × split).
export CASPER_DATA_ROOT="$(pwd)/data_sample"

# Stage 1 — entity extraction + semantic compression for both splits.
CASPER_STAGE1_SPLIT=train ./scripts/run_stage1.sh
CASPER_STAGE1_SPLIT=test  ./scripts/run_stage1.sh

# Stage 2 — embed compressed text into ChromaDB.
./scripts/run_stage2.sh

# Stage 3 — retrieve top-K nearest train docs for each test query.
# With only 5 train docs we cap K at 5 (the default sweep is 50).
CASPER_STAGE3_TOP_K=5 ./scripts/run_stage3.sh

# Stage 4 — classify with retrieved compressed exemplars.
CASPER_STAGE4_TOP_K=5 ./scripts/run_stage4.sh
```

The sample dataset under `data_sample/{train,test}/{paper,patent,report}_{train,test}.json`
contains 5 documents per file (deterministically sampled, seed 42) following
the schema `{doc_id: {CN, text, classes_top, classes_sub}}`, where
`classes_top` / `classes_sub` are `[{class_id, class_label}, ...]` lists.
The full benchmark is not included in this repo — drop your own corpus into
`data/{train,test}/` (or any directory pointed to by `CASPER_DATA_ROOT`)
following the same schema.

## Data schema

Each split JSON is a mapping `{doc_id: record}`, where each record has:

| Field | Type | Notes |
|---|---|---|
| `text` | string | Raw document text (title + abstract for papers; abstract / claim text for patents; project summary for reports). May be English or Korean. |
| `classes_top` | `[{class_id: int, class_label: str}, ...]` | First-level (coarse) taxonomy labels. |
| `classes_sub` | `[{class_id: int, class_label: str}, ...]` | Second-level (fine-grained) labels — the primary evaluation target. |
| `CN` | string \| null | Source identifier (optional). |

Papers and patents are multi-label; reports are single-label (single entry in
`classes_sub`). Fine-grained label spaces in the reference benchmark are on
the order of 100 / 139 / 88 categories for paper / patent / report.

## Models

Both stages of the compressor and the final classifier are served by **one
generative LLM** loaded with vLLM. The expected backbone is a 4 B
instruction-tuned sLLM supervised-fine-tuned on the CASPER instruction set
covering entity extraction, semantic compression, and classification —
point `CASPER_GEN_MODEL` at your own HuggingFace repository id. Embedding is
handled by **`Qwen/Qwen3-Embedding-4B`** via `sentence-transformers`
(override with `CASPER_EMBED_MODEL`). The ChromaDB index uses HNSW for
approximate nearest-neighbor search.

To swap in a different base LLM for ablation, set
`CASPER_GEN_MODEL=<hf-repo-id>` before launching any stage.

## Configuration

All knobs live in `src/casper/config.py` and can be overridden via `CASPER_*`
environment variables. Common ones:

| Env var | Default | Used by |
|---|---|---|
| `CASPER_DATA_ROOT` | `<project>/data` | Stage 1 input root (point at `data_sample` for smoke runs) |
| `CASPER_GEN_MODEL` | _(set to your fine-tuned CASPER-4B HF repo id)_ | Stages 1 & 4 (compression + classification) |
| `CASPER_EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B` | Stage 2 |
| `CASPER_RATIO` | `0.2` | Stage 1 compression ratio `r = m / M` (default 20 %) |
| `CASPER_RUN_NAME` | `<gen_short>_ratio_<RATIO>` | All stages — output subdir |
| `CASPER_STAGE1_SPLIT` | `test` | Stage 1 — `train` or `test` |
| `CASPER_STAGE1_SUMMARY_SKIP` | `false` | Stage 1 — keyword extraction only, skip rewriting |
| `CASPER_STAGE2_SPLITS` | `train,test` | Stage 2 |
| `CASPER_STAGE2_SOURCE` | `both` | Stage 2 input — `text`, `summary`, or `both` |
| `CASPER_STAGE3_TOP_K` | `50` | Stage 3 retrieval depth |
| `CASPER_STAGE3_SOURCE` | `summary` | Stage 3 — which Stage-2 collection to pair (`summary` or `text`) |
| `CASPER_STAGE4_TOP_K` | `5,10,15,20,25,30,40,50` | Stage 4 — sweep of in-context exemplar counts |
| `CASPER_STAGE4_CLASS_TYPES` | `sub,top` | Stage 4 — taxonomy levels to evaluate |
| `CASPER_CUDA_VISIBLE_DEVICES` | `0` | All GPU stages |
| `CASPER_VLLM_TP` | `1` | vLLM tensor-parallel size (`2` for dual-GPU sharding) |
| `CASPER_STAGE4_BATCH_SIZE` | `128` | Stage 4 — prompts per `model.generate` call |

The recommended operating point is `CASPER_RATIO=0.2` (top-20 % of extracted
entities) and `CASPER_STAGE4_TOP_K=20` for fine-grained classification.

## No-compression baseline

To reproduce the *Original (no compression)* baseline — feeding raw retrieved
documents to the classifier instead of the entity-centric summary — point
Stage 2 and Stage 3 at the raw `text` source and rerun Stage 4 on the
resulting retrieval:

```bash
CASPER_STAGE2_SOURCE=text ./scripts/run_stage2.sh
CASPER_STAGE3_SOURCE=text ./scripts/run_stage3.sh
./scripts/run_stage4.sh
```

Stage-4 results land in a `text/` subdirectory of the run, so the compressed
and uncompressed runs coexist under the same `RUN_NAME`.

## Multi-GPU throughput

For a full-scale run, two GPUs and batched generation cut wall time
substantially:

```bash
export CASPER_CUDA_VISIBLE_DEVICES=0,1
export CASPER_VLLM_TP=2              # shard the 4B sLLM across both GPUs
export CASPER_STAGE4_BATCH_SIZE=128  # prompts per vLLM generate() call
```

- **TP=2 vLLM** for Stage 1 + Stage 4.
- **Batched generation** in Stage 4: `STAGE4_BATCH_SIZE` prompts per
  `model.generate` call.
- **Batched ChromaDB queries** in Stage 3: one `train_col.query` per fetch
  batch.

## Notes

- Stage 2 defaults to **append** mode; set `CASPER_STAGE2_OVERWRITE=true` to
  rebuild collections from scratch.
- Model weights are pulled into `model_cache/` (HuggingFace hub layout).
  `config.py` sets `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` /
  `HF_HUB_CACHE` to that path on import, and Stage 1/4 also pass
  `download_dir` to vLLM explicitly — no model files end up in
  `~/.cache/huggingface`. `model_cache/` is gitignored.

## License

MIT — see [LICENSE](LICENSE).
