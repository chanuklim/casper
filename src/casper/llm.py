"""vLLM helpers shared between Stage 1 (keyword/summary) and Stage 4 (classify).

The two stages tune sampling parameters very differently, so the exact param
sets stay with each stage. This module provides:

- :func:`load_generator` — instantiate vLLM + tokenizer with stage-specific args.
- :func:`apply_chat_template` — wrap a raw user prompt in the chat template.
- :class:`SamplingParamsBank` — hold a primary :class:`SamplingParams` and an
  ordered list of fallbacks; :meth:`for_attempt` picks the right one per retry.
- :func:`generate_with_retry` — retry generation until ``validate`` accepts the
  output, escalating sampling params each attempt.
- :func:`print_gpu_memory` — quick diagnostic.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_generator(
    model_name: str,
    *,
    cuda_visible_devices: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: Optional[int] = None,
    tensor_parallel_size: int = 1,
    download_dir: Optional[str] = None,
) -> Tuple[LLM, AutoTokenizer]:
    """Initialize vLLM + tokenizer with the given resource budget.

    ``download_dir`` is forwarded to vLLM and used as the tokenizer ``cache_dir``
    so weights/tokenizer files are stored alongside everything else under
    :data:`config.MODEL_CACHE_DIR` instead of the global HF cache.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=download_dir,
    )

    llm_kwargs = dict(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if download_dir is not None:
        llm_kwargs["download_dir"] = download_dir

    model = LLM(**llm_kwargs)
    return model, tokenizer


def apply_chat_template(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def apply_chat_template_batch(prompts: Sequence[str], tokenizer) -> List[str]:
    return [apply_chat_template(p, tokenizer) for p in prompts]


@dataclass
class SamplingParamsBank:
    """Primary + fallback sampling params for retry-driven generation."""

    primary: SamplingParams
    alternatives: List[SamplingParams] = field(default_factory=list)
    dynamic_factory: Optional[Callable[[int], SamplingParams]] = None

    def for_attempt(self, attempt: int) -> SamplingParams:
        if attempt == 0:
            return self.primary
        idx = attempt - 1
        if idx < len(self.alternatives):
            return self.alternatives[idx]
        if self.dynamic_factory is not None:
            return self.dynamic_factory(attempt)
        # Fall back to last alternative (or primary if no alternatives).
        return self.alternatives[-1] if self.alternatives else self.primary


def generate_with_retry(
    model: LLM,
    tokenizer,
    prompt: str,
    bank: SamplingParamsBank,
    *,
    max_retries: int,
    retry_delay: float,
    validate: Optional[Callable[[str], Tuple[bool, str]]] = None,
    log_prefix: str = "        ",
) -> Tuple[Optional[str], int]:
    """Generate ``prompt`` until ``validate`` accepts the output.

    Returns ``(text, attempts_used)``. If every attempt fails validation,
    returns ``(None, max_retries)``.
    """
    formatted = apply_chat_template(prompt, tokenizer)

    for attempt in range(max_retries):
        params = bank.for_attempt(attempt)
        try:
            outputs = model.generate([formatted], params)
            text = outputs[0].outputs[0].text

            if validate is None:
                return text, attempt + 1

            ok, message = validate(text)
            if ok:
                return text, attempt + 1
            print(f"{log_prefix}Attempt {attempt + 1} failed: {message}")
        except Exception as exc:  # noqa: BLE001 — vLLM raises a wide variety
            print(f"{log_prefix}Attempt {attempt + 1} error: {exc}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    return None, max_retries


def print_gpu_memory(prefix: str = "    ") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
