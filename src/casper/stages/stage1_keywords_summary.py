"""Stage 1 — keyword extraction + category-driven summarization with vLLM.

Reads JSON files in ``data/<split>/`` (where each value has ``text`` and
``classes_sub``), generates per-document ``response_keywords`` (7-category
JSON) and ``response_summ`` (summary of length proportional to the keyword
count via :data:`config.RATIO`), and writes results to
``outputs/stage1_keywords_summary/<run>/<split>/``.

Resumable: if the output file already exists, items that have both
``response_keywords`` and ``response_summ`` are skipped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

from vllm import SamplingParams

from casper import config
from casper.json_utils import (
    count_keywords,
    empty_keyword_json,
    extract_json_from_response,
    extract_summary_response,
    keyword_json_score,
    validate_keyword_json,
)
from casper.llm import (
    SamplingParamsBank,
    apply_chat_template_batch,
    load_generator,
    print_gpu_memory,
)
from casper.prompts import (
    KEYWORD_PROMPT,
    KEYWORD_PROMPT_TEST,
    SUMMARY_PROMPT,
)


KEYWORD_PRIMARY = SamplingParams(
    temperature=0.1,
    top_p=0.1,
    repetition_penalty=1.1,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    max_tokens=2048,
)

KEYWORD_ALTERNATIVES = [
    SamplingParams(
        temperature=0.5, top_p=0.9, repetition_penalty=1.05,
        frequency_penalty=0.05, presence_penalty=0.05, max_tokens=2048,
    ),
    SamplingParams(
        temperature=0.4, top_p=0.8, repetition_penalty=1.1,
        frequency_penalty=0.15, presence_penalty=0.15, max_tokens=2048,
    ),
    SamplingParams(
        temperature=0.2, top_p=0.5, repetition_penalty=1.15,
        frequency_penalty=0.2, presence_penalty=0.2, max_tokens=1536,
    ),
]


def _dynamic_keyword_params(attempt: int) -> SamplingParams:
    return SamplingParams(
        temperature=min(0.7, 0.1 + 0.1 * attempt),
        top_p=min(0.95, 0.3 + 0.1 * attempt),
        repetition_penalty=max(1.0, 1.2 - 0.05 * attempt),
        frequency_penalty=max(0.0, 0.3 - 0.05 * attempt),
        presence_penalty=max(0.0, 0.3 - 0.05 * attempt),
        max_tokens=2048,
    )


KEYWORD_BANK = SamplingParamsBank(
    primary=KEYWORD_PRIMARY,
    alternatives=KEYWORD_ALTERNATIVES,
    dynamic_factory=_dynamic_keyword_params,
)


def _generate_keywords_batch(model, tokenizer, prompts: List[str]) -> List[str]:
    """Batched keyword generation with retry escalation.

    Each retry round submits *all* still-pending prompts in one ``model.generate``
    call so vLLM can fuse them into a single optimized batch. Items that pass
    validation drop out of subsequent rounds; items that don't escalate to the
    next sampling preset. The semantics match the previous per-item loop:
    accept the first response with ``score >= STAGE1_KEYWORD_MIN_NONEMPTY``,
    otherwise keep the best-scoring response, and fall back to an empty JSON
    structure only if no attempt produced a valid one.
    """
    n = len(prompts)
    if n == 0:
        return []

    formatted = apply_chat_template_batch(prompts, tokenizer)

    best_results: List[Optional[str]] = [None] * n
    best_scores: List[int] = [0] * n
    pending: List[int] = list(range(n))

    print(f"      Batched keyword generation: {n} items, max_retries={config.STAGE1_MAX_RETRIES}")
    print_gpu_memory()

    accepted_at: List[int] = [0] * n

    for attempt in range(config.STAGE1_MAX_RETRIES):
        if not pending:
            break

        params = KEYWORD_BANK.for_attempt(attempt)
        batch_prompts = [formatted[i] for i in pending]

        try:
            outputs = model.generate(batch_prompts, params)
        except Exception as exc:  # noqa: BLE001
            print(f"      Attempt {attempt + 1} batched generate failed: {exc}")
            time.sleep(config.STAGE1_RETRY_DELAY)
            continue

        next_pending: List[int] = []
        accepted = 0
        improved = 0

        for k, idx in enumerate(pending):
            try:
                text = outputs[k].outputs[0].text
                json_str = extract_json_from_response(text)
                ok, _msg = validate_keyword_json(json_str, min_nonempty=3)
            except Exception:  # noqa: BLE001
                ok = False
                json_str = None

            if ok and json_str is not None:
                score = keyword_json_score(json_str)
                if score > best_scores[idx]:
                    best_results[idx] = json_str
                    best_scores[idx] = score
                    improved += 1
                if score >= config.STAGE1_KEYWORD_MIN_NONEMPTY:
                    accepted += 1
                    accepted_at[idx] = attempt + 1
                    continue

            next_pending.append(idx)

        print(
            f"      Attempt {attempt + 1}: accepted {accepted}, "
            f"improved {improved}, retrying {len(next_pending)}"
        )

        pending = next_pending
        if pending and attempt < config.STAGE1_MAX_RETRIES - 1:
            time.sleep(config.STAGE1_RETRY_DELAY)

    # Anything still pending uses its best-so-far result, or empty JSON if none.
    fallback_empty = 0
    fallback_partial = 0
    for idx in range(n):
        if best_results[idx] is None:
            best_results[idx] = empty_keyword_json()
            fallback_empty += 1
        elif accepted_at[idx] == 0:
            fallback_partial += 1

    if fallback_empty or fallback_partial:
        print(
            f"      Final: {n - fallback_empty - fallback_partial} accepted, "
            f"{fallback_partial} partial (best score < min), {fallback_empty} empty fallback"
        )

    return [r for r in best_results]  # type: ignore[return-value]


def _generate_summaries_batch(model, tokenizer, prompts: List[str]) -> List[Optional[str]]:
    try:
        formatted = apply_chat_template_batch(prompts, tokenizer)
        outputs = model.generate(formatted, KEYWORD_PRIMARY)
        return [extract_summary_response(o.outputs[0].text) for o in outputs]
    except Exception as exc:  # noqa: BLE001
        print(f"Error in summary generation: {exc}")
        return [None] * len(prompts)


def _process_file(
    model, tokenizer,
    input_path: Path, output_path: Path,
    ratio: float, batch_size: int,
    split: str,
) -> Tuple[bool, str]:
    """Generate keywords + summaries for one JSON file. Resumable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_path.exists():
        print("  Loading existing output file for resuming...")
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    modified = False
    save_counter = 0

    batch_items: list = []
    batch_keyword_prompts: List[Optional[str]] = []
    batch_keys: list = []

    total_items = len(data)
    processed_count = 0

    def _flush(final: bool = False) -> None:
        nonlocal modified, save_counter
        if not batch_items:
            return

        # Step 1: keywords
        kw_to_generate = [p for p in batch_keyword_prompts if p is not None]
        if kw_to_generate:
            print(f"    Generating keywords for {len(kw_to_generate)} items...")
            kw_batch = _generate_keywords_batch(model, tokenizer, kw_to_generate)
            kw_idx = 0
            for idx, prompt in enumerate(batch_keyword_prompts):
                if prompt is None:
                    continue
                if kw_idx < len(kw_batch) and kw_batch[kw_idx]:
                    batch_items[idx]["response_keywords"] = kw_batch[kw_idx]
                    modified = True
                else:
                    print(f"      Warning: no keywords generated for item {batch_keys[idx]}")
                kw_idx += 1

        # Step 2: summaries (for items with keywords but no summary)
        if not config.STAGE1_SUMMARY_SKIP:
            summ_prompts: List[str] = []
            summ_indices: List[int] = []
            for idx, item in enumerate(batch_items):
                if "response_keywords" not in item or "response_summ" in item:
                    continue
                total_kw = count_keywords(item["response_keywords"])
                max_items_local = max(1, int(total_kw * ratio))
                summ_prompt = SUMMARY_PROMPT.format(
                    max_items=max_items_local,
                    total_items=total_kw,
                    text=item.get("text", ""),
                    categories=item.get("response_keywords", ""),
                )
                item["prompt_summ"] = summ_prompt
                summ_prompts.append(summ_prompt)
                summ_indices.append(idx)

            if summ_prompts:
                print(f"    Generating summaries for {len(summ_prompts)} items...")
                summ_batch = _generate_summaries_batch(model, tokenizer, summ_prompts)
                for s_i, item_idx in enumerate(summ_indices):
                    if s_i < len(summ_batch) and summ_batch[s_i]:
                        batch_items[item_idx]["response_summ"] = summ_batch[s_i]
                        modified = True
                    else:
                        print(f"      Warning: no summary generated for item {batch_keys[item_idx]}")

        if modified:
            save_counter += 1
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            label = "Final" if final else "Intermediate"
            print(f"    ✓ {label} save #{save_counter}")

        batch_items.clear()
        batch_keyword_prompts.clear()
        batch_keys.clear()

    try:
        for key, item in data.items():
            processed_count += 1

            already_done = "response_keywords" in item and (
                config.STAGE1_SUMMARY_SKIP or "response_summ" in item
            )
            if already_done:
                print(f"    Item {key}: already complete, skipping...")
                continue

            text = item.get("text", "")

            if "response_keywords" not in item:
                if split == "test":
                    kw_prompt = KEYWORD_PROMPT_TEST.format(abstract=text)
                else:
                    class_labels = [
                        cls["class_label"]
                        for cls in item.get("classes_sub", [])
                        if "class_label" in cls
                    ]
                    classlabel = ", ".join(class_labels)
                    kw_prompt = KEYWORD_PROMPT.format(abstract=text, classlabel=classlabel)
                item["prompt_keywords"] = kw_prompt
                batch_keyword_prompts.append(kw_prompt)
            else:
                batch_keyword_prompts.append(None)

            batch_items.append(item)
            batch_keys.append(key)

            if len(batch_items) >= batch_size:
                print(f"  Processing batch of {len(batch_items)} items... ({processed_count}/{total_items})")
                _flush(final=False)

        if batch_items:
            print(f"  Processing final batch of {len(batch_items)} items...")
            _flush(final=True)

        if modified:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True, f"Successfully processed with {save_counter} saves: {output_path}"
        return False, f"No new items to process in: {input_path}"

    except Exception as exc:  # noqa: BLE001
        if modified:
            emergency = output_path.with_suffix(".emergency.json")
            with open(emergency, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"    ⚠️ Emergency save to: {emergency}")
        return False, f"Error processing {input_path}: {exc}"


def main() -> None:
    config.ensure_dirs()
    split = config.STAGE1_SPLIT
    input_dir = config.stage1_input_dir(split)
    output_dir = config.stage1_output_dir(split)

    print("Initializing vLLM model...")
    print(f"Using model: {config.GEN_MODEL}")
    print(f"Run name: {config.RUN_NAME}")
    print(f"Split: {split}")
    print(f"Ratio: {config.RATIO}")
    print(f"Max retries for JSON generation: {config.STAGE1_MAX_RETRIES}")
    print(f"Summary skip:    {config.STAGE1_SUMMARY_SKIP}")
    print("=" * 50)

    model, tokenizer = load_generator(
        config.GEN_MODEL,
        cuda_visible_devices=config.CUDA_VISIBLE_DEVICES,
        gpu_memory_utilization=config.VLLM_GPU_MEMORY_UTILIZATION_STAGE1,
        max_model_len=config.VLLM_MAX_MODEL_LEN_STAGE1,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        max_num_batched_tokens=config.VLLM_MAX_NUM_BATCHED_TOKENS,
        tensor_parallel_size=config.VLLM_TENSOR_PARALLEL_SIZE,
        download_dir=str(config.MODEL_CACHE_DIR),
    )

    print("\nTesting model with a simple prompt...")
    test_prompt = apply_chat_template_batch(["Hello, can you hear me?"], tokenizer)[0]
    test_output = model.generate([test_prompt], KEYWORD_PRIMARY)
    print(f"Test response: {test_output[0].outputs[0].text[:100]}...")
    print("=" * 50)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    print(f"\nFound {len(json_files)} JSON files in {input_dir}")

    total_files = 0
    processed_files = 0
    errors: List[str] = []

    for json_file in json_files:
        total_files += 1
        print(f"\n{'=' * 30}\nFile {total_files}: {json_file.name}\n{'=' * 30}")
        output_path = output_dir / json_file.name
        print(f"Output path: {output_path}")

        success, message = _process_file(
            model, tokenizer,
            json_file, output_path,
            config.RATIO, config.STAGE1_BATCH_SIZE,
            split,
        )
        if success:
            processed_files += 1
            print(f"✓ Completed: {message}")
        else:
            errors.append(message)
            print(f"✗ Failed: {message}")

    print(f"\n{'=' * 50}\nSummary\n{'=' * 50}")
    print(f"Total files attempted: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {len(errors)}")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    main()
