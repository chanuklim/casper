"""Stage 4 — multi-label classification with retrieved examples.

For each retrieved-data file × class_type × top-k value, generate the LLM
classification answer with vLLM and accumulate metrics. Resumable per output
file (skips items whose ``response`` is already saved).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from vllm import SamplingParams

from casper import config
from casper.json_utils import parse_classification_answer
from casper.llm import (
    SamplingParamsBank,
    apply_chat_template,
    load_generator,
)
from casper.metrics import MetricsAccumulator, calculate_accuracy
from casper.prompts import CLASSIFY_PROMPT, CLASSIFY_PROMPT_PATENT


CLASSIFY_PRIMARY = SamplingParams(
    temperature=0.1,
    top_p=0.1,
    repetition_penalty=1.1,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    max_tokens=4096,
    stop=["</answer>"],
)

CLASSIFY_ALTERNATIVES = [
    SamplingParams(
        temperature=0.2, top_p=0.5, repetition_penalty=1.1,
        frequency_penalty=0.15, presence_penalty=0.15, max_tokens=4096,
        stop=["</answer>"],
    ),
    SamplingParams(
        temperature=0.4, top_p=0.8, repetition_penalty=1.15,
        frequency_penalty=0.2, presence_penalty=0.2, max_tokens=4096,
        stop=["</answer>"],
    ),
    SamplingParams(
        temperature=0.5, top_p=0.9, repetition_penalty=1.05,
        frequency_penalty=0.05, presence_penalty=0.05, max_tokens=4096,
        stop=["</answer>"],
    ),
]

CLASSIFY_BANK = SamplingParamsBank(
    primary=CLASSIFY_PRIMARY, alternatives=CLASSIFY_ALTERNATIVES
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_json_data(file_path: Path) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_data(data: List[Dict], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_retrieved_items(retrieved_items: List[Dict], k: int, class_type_str: str) -> str:
    items_text: List[str] = []
    for idx, item in enumerate(retrieved_items[:k], 1):
        classes_key = f"classes_{class_type_str}"
        classes = item.get(classes_key, []) or []
        labels = [f"{cls['class_label']}({cls['class_id']})" for cls in classes]
        class_text = ", ".join(labels) if labels else "No class information"

        formatted = (
            f"Top-{idx}. Retrieved Item ID {item['id']}: \n"
            f"- Text: {item.get('document', 'No text available')}\n"
            f"- Class label(ID): {class_text}\n"
        )
        items_text.append(formatted)
    return "\n".join(items_text)


def create_prompt(data_item: Dict, k: int, class_type_str: str, is_patent: bool) -> str:
    template = CLASSIFY_PROMPT_PATENT if is_patent else CLASSIFY_PROMPT
    retrieved_text = format_retrieved_items(data_item["retrieved"], k, class_type_str)
    return template.format(
        target_id=data_item["query_id"],
        target_text=data_item["query_text"],
        retrieved_count=k,
        retrieved_items_text=retrieved_text,
    )


def run_inference_with_retry(
    model, tokenizer, prompt: str, max_retries: int = config.STAGE4_MAX_RETRIES
) -> Optional[str]:
    """Single-prompt path kept for parity with prior callers / debug scripts.
    The batched path (``classify_chunk``) is preferred for production runs."""
    formatted = apply_chat_template(prompt, tokenizer)

    for attempt in range(max_retries):
        params = CLASSIFY_BANK.for_attempt(attempt)
        try:
            outputs = model.generate([formatted], params)
            response = outputs[0].outputs[0].text

            if "</answer>" not in response and "<answer>" in response:
                response = response + "</answer>"

            if "<answer>" in response and "</answer>" in response:
                return response

            print(f"Attempt {attempt + 1}: invalid response format, retrying...")
            print(f"  Response preview: {response[:300]}...")
            time.sleep(config.STAGE4_RETRY_DELAY)
        except Exception as exc:  # noqa: BLE001
            print(f"Attempt {attempt + 1} failed with error: {exc}")
            if attempt < max_retries - 1:
                time.sleep(config.STAGE4_RETRY_DELAY)
            else:
                print("Max retries reached. Returning None.")
                return None

    return None


def classify_chunk(
    model, tokenizer,
    items: List[Dict], k: int, class_type: str, is_patent: bool,
    max_retries: int = config.STAGE4_MAX_RETRIES,
) -> List[Optional[str]]:
    """Batched classification for one chunk.

    All prompts are sent to vLLM in one ``generate`` call; items whose
    output fails the ``<answer>...</answer>`` format check are collected
    and retried with the next sampling-param tier (sampled per attempt
    via ``CLASSIFY_BANK``).
    """
    prompts = [create_prompt(it, k, class_type, is_patent) for it in items]
    formatted = [apply_chat_template(p, tokenizer) for p in prompts]
    n = len(formatted)
    pending: List[int] = list(range(n))
    results: List[Optional[str]] = [None] * n

    for attempt in range(max_retries):
        if not pending:
            break
        params = CLASSIFY_BANK.for_attempt(attempt)
        cur_prompts = [formatted[i] for i in pending]
        try:
            outs = model.generate(cur_prompts, params)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! batched attempt {attempt + 1} failed: {exc}")
            time.sleep(config.STAGE4_RETRY_DELAY)
            continue

        new_pending: List[int] = []
        for j, idx in enumerate(pending):
            text = outs[j].outputs[0].text
            if "</answer>" not in text and "<answer>" in text:
                text = text + "</answer>"
            if "<answer>" in text and "</answer>" in text:
                results[idx] = text
            else:
                new_pending.append(idx)
        succeeded = len(pending) - len(new_pending)
        if new_pending:
            print(f"  attempt {attempt + 1}: ok={succeeded}, retry={len(new_pending)}")
        else:
            print(f"  attempt {attempt + 1}: ok={succeeded}")
        pending = new_pending

    return results


def main() -> None:
    config.ensure_dirs()
    set_random_seed(config.STAGE4_RANDOM_SEED)

    print(f"Loading generator: {config.GEN_MODEL}  (TP={config.VLLM_TENSOR_PARALLEL_SIZE})")
    model, tokenizer = load_generator(
        config.GEN_MODEL,
        cuda_visible_devices=config.CUDA_VISIBLE_DEVICES,
        gpu_memory_utilization=config.VLLM_GPU_MEMORY_UTILIZATION_STAGE4,
        max_model_len=config.VLLM_MAX_MODEL_LEN_STAGE4,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        tensor_parallel_size=config.VLLM_TENSOR_PARALLEL_SIZE,
        download_dir=str(config.MODEL_CACHE_DIR),
    )

    input_dir = config.stage3_output_dir()
    output_root = config.stage4_output_dir()
    if not input_dir.exists():
        raise SystemExit(f"Stage-3 output not found: {input_dir}. Run Stage 3 first.")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No retrieved JSON files in {input_dir}.")

    print(f"Found {len(json_files)} retrieved-data files in {input_dir}")
    print(f"Top-K values: {config.STAGE4_TOP_K}")
    print(f"Class types:  {config.STAGE4_CLASS_TYPES}")

    experiment_results: Dict[str, Dict] = {}

    for ct in config.STAGE4_CLASS_TYPES:
        for k in config.STAGE4_TOP_K:
            print(f"\n{'=' * 60}\nProcessing: class_type={ct}, k={k}\n{'=' * 60}")

            for json_file in json_files:
                is_patent = "patent" in json_file.name.lower()
                output_file = output_root / ct / f"k_{k}" / f"results_{json_file.stem}.json"

                data = load_json_data(json_file)

                results = load_json_data(output_file) if output_file.exists() else []
                if results:
                    print(f"Loaded existing results: {len(results)} items")

                if len(data) > config.STAGE4_MAX_DATA_NUM:
                    np.random.seed(config.STAGE4_RANDOM_SEED)
                    sampled_indices = np.random.choice(
                        len(data), config.STAGE4_MAX_DATA_NUM, replace=False
                    )
                    sampled_data = [data[i] for i in sampled_indices]
                else:
                    sampled_data = data

                print(f"Processing {len(sampled_data)} items from {json_file.name}  "
                      f"(batch={config.STAGE4_BATCH_SIZE})")

                metrics = MetricsAccumulator()

                # Phase A: ingest already-completed items into the metric accumulator.
                done_ids: set = set()
                for r in results:
                    if "response" in r:
                        done_ids.add(r["question_id"])
                        if "response_classes" in r:
                            metrics.add_sample(r["true_classes"], r["response_classes"])
                if done_ids:
                    cur = metrics.get_all_metrics()
                    print(f"  Resume: {len(done_ids)} already done. "
                          f"EM={cur['exact_match']:.3f} F1-micro={cur['f1_micro']:.3f} "
                          f"F1-macro={cur['f1_macro']:.3f}")

                # Phase B: batched generation over remaining items.
                todo = [it for it in sampled_data if it["query_id"] not in done_ids]
                bsz = max(1, config.STAGE4_BATCH_SIZE)
                n_chunks = (len(todo) + bsz - 1) // bsz
                save_after = max(1, config.STAGE4_SAVE_INTERVAL)

                for chunk_idx, start in enumerate(range(0, len(todo), bsz), start=1):
                    chunk = todo[start:start + bsz]
                    print(f"  Chunk {chunk_idx}/{n_chunks}: {len(chunk)} items "
                          f"({start + 1}–{start + len(chunk)} of {len(todo)})")
                    responses = classify_chunk(model, tokenizer, chunk, k, ct, is_patent)

                    for item, response in zip(chunk, responses):
                        if response is None:
                            print(f"    ! failed: {item['query_id']}")
                            continue
                        parsed = parse_classification_answer(response)
                        result_item = {
                            "question_id": item["query_id"],
                            "question": item["query_text"],
                            "true_classes": item[f"classes_{ct}"],
                            "response": response,
                            "response_classes": parsed if parsed is not None else [],
                        }
                        existing_idx = next(
                            (i for i, r in enumerate(results) if r["question_id"] == result_item["question_id"]),
                            None,
                        )
                        if existing_idx is not None:
                            results[existing_idx] = result_item
                        else:
                            results.append(result_item)
                        if parsed is not None:
                            metrics.add_sample(result_item["true_classes"], parsed)

                    if chunk_idx % save_after == 0 or chunk_idx == n_chunks:
                        save_json_data(results, output_file)
                        cur = metrics.get_all_metrics()
                        if cur["total_samples"] > 0:
                            print(f"    saved ({len(results)} total). "
                                  f"EM={cur['exact_match']:.3f} F1-micro={cur['f1_micro']:.3f} "
                                  f"F1-macro={cur['f1_macro']:.3f}")

                save_json_data(results, output_file)

                final_metrics = metrics.get_all_metrics()
                if final_metrics["total_samples"] > 0:
                    key = f"{json_file.stem}_{ct}_k{k}"
                    experiment_results[key] = {
                        "file": json_file.name,
                        "class_type": ct,
                        "k": k,
                        "processed_count": final_metrics["total_samples"],
                        "metrics": final_metrics,
                    }
                    print(f"\n{'=' * 50}")
                    print(f"FINAL RESULTS — {json_file.name} (k={k}, class={ct})")
                    print(f"{'=' * 50}")
                    print(f"Total processed:    {final_metrics['total_samples']}")
                    print(f"Exact Match:        {final_metrics['exact_match']:.4f}")
                    print(f"Precision/Recall/F1 (sample-avg): "
                          f"{final_metrics['precision_avg']:.4f} / "
                          f"{final_metrics['recall_avg']:.4f} / "
                          f"{final_metrics['f1_avg']:.4f}")
                    print(f"F1-micro: {final_metrics['f1_micro']:.4f}")
                    print(f"F1-macro: {final_metrics['f1_macro']:.4f}")

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print(f"\n{'File':<25} {'Class':<5} {'k':<3} {'N':<5} {'EM':<7} {'F1-avg':<7} {'F1-micro':<9} {'F1-macro':<9}")
    print("-" * 80)
    for key in sorted(experiment_results.keys()):
        result = experiment_results[key]
        m = result["metrics"]
        file_short = result["file"][:20] + "..." if len(result["file"]) > 20 else result["file"]
        print(
            f"{file_short:<25} {result['class_type']:<5} {result['k']:<3} "
            f"{result['processed_count']:<5} "
            f"{m['exact_match']:<7.4f} "
            f"{m['f1_avg']:<7.4f} "
            f"{m['f1_micro']:<9.4f} "
            f"{m['f1_macro']:<9.4f}"
        )

    summary_file = output_root / "experiment_summary_with_f1_metrics.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)
    print(f"\nExperiment summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
