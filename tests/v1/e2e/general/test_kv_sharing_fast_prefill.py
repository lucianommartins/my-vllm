# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import random

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from ....utils import check_answers, fork_new_process_for_each_test, prep_prompts

# global seed
SEED = 42


@pytest.fixture
def test_prompts():
    """Adapted from tests/v1/e2e/spec_decode/utils.py."""
    prompt_types = ["repeat", "sentence"]
    # Setting higher num prompts increases the chance of numerics mismatch
    # due to matrix multiplication numerics depending on batch dimension
    num_prompts = 10
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        if kind == "repeat":
            prompt = f"""please repeat the word '{word}' 10 times."""
        elif kind == "sentence":
            prompt = f"""please give a ten-word sentence that
            uses the word {word} at least once."""
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append(prompt)

    return prompts


use_fork_for_test = (
    fork_new_process_for_each_test
    if not (current_platform.is_rocm() or current_platform.is_xpu())
    else lambda x: x
)


@use_fork_for_test
@pytest.mark.parametrize(
    "kv_sharing_fast_prefill,enforce_eager",
    [
        (True, True),
        (True, False),
        # (False, False) omitted: it saves ~9 min of mostly torch.compile time
        # in CI.
        (False, True),
    ],
)
def test_kv_sharing_fast_prefill(
    monkeypatch: pytest.MonkeyPatch,
    kv_sharing_fast_prefill: bool,
    enforce_eager: bool,
):
    if not enforce_eager and current_platform.is_rocm():
        # Relevant context: https://github.com/vllm-project/vllm/pull/29244
        pytest.skip(
            "ROCm: torch.compile produces incorrect output for gemma-3n's GELU "
            "with tanh approximation. Use enforce_eager=True instead."
        )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE
        if not enforce_eager
        else CompilationMode.NONE,
    )
    batch_size = 10

    with monkeypatch.context() as m:
        # Make scheduling deterministic for reproducibility
        if current_platform.is_rocm():
            # Use spawn to prevent cuda re-initialization error
            m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        else:
            m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        prompts, answer, indices = prep_prompts(batch_size)

        llm = LLM(
            model="google/gemma-3n-E2B-it",
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
            seed=SEED,
            kv_sharing_fast_prefill=kv_sharing_fast_prefill,
            attention_backend="TRITON_ATTN",
        )
        responses = llm.generate(prompts, sampling_params)
        check_answers(
            indices,
            answer,
            [response.outputs[0].text for response in responses],
            accept_rate=1.0,
        )


@use_fork_for_test
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_gemma4_kv_sharing_fast_prefill_parity(
    monkeypatch: pytest.MonkeyPatch,
    enforce_eager: bool,
):
    if not enforce_eager and current_platform.is_rocm():
        pytest.skip("ROCm: torch.compile on eager-only fast prefill CI configuration.")

    long_context = " ".join(f"item_{i}" for i in range(140))
    prompts = [
        (
            "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        (
            f"<start_of_turn>user\nSummarize the list: {long_context}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "<start_of_turn>user\nCompute 17 + 25.<end_of_turn>\n<start_of_turn>model\n",
    ]
    gen_params = SamplingParams(temperature=0.0, max_tokens=32)
    logprob_params = SamplingParams(temperature=0.0, max_tokens=4, prompt_logprobs=1)

    with monkeypatch.context() as m:
        if current_platform.is_rocm():
            m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        else:
            m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        baseline_llm = LLM(
            model="google/gemma-4-E2B-it",
            enforce_eager=enforce_eager,
            seed=SEED,
            kv_sharing_fast_prefill=False,
            max_model_len=1024,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.4,
        )
        try:
            baseline_outputs = [
                r.outputs[0].text for r in baseline_llm.generate(prompts, gen_params)
            ]
            baseline_logprobs = baseline_llm.generate(prompts, logprob_params)
        finally:
            baseline_llm.llm_engine.engine_core.shutdown()
            del baseline_llm
            gc.collect()
            cleanup_dist_env_and_memory()
            torch.accelerator.empty_cache()

        fast_llm = LLM(
            model="google/gemma-4-E2B-it",
            enforce_eager=enforce_eager,
            seed=SEED,
            max_model_len=1024,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.4,
        )
        try:
            assert fast_llm.llm_engine.vllm_config.cache_config.kv_sharing_fast_prefill
            fast_outputs = [
                r.outputs[0].text for r in fast_llm.generate(prompts, gen_params)
            ]
            assert fast_outputs == baseline_outputs

            # enforce_eager=True executes identical eager layers; enforce_eager=False
            # compares 1 vs 2 Inductor FX graphs on tail bf16 logprobs, where a
            # rank-1 tie swap can change which non-prompt token appears in top-1.
            fast_logprobs = fast_llm.generate(prompts, logprob_params)
            for base_r, fast_r in zip(baseline_logprobs, fast_logprobs):
                assert base_r.prompt_token_ids is not None
                assert base_r.prompt_logprobs is not None
                assert fast_r.prompt_logprobs is not None
                assert len(base_r.prompt_logprobs) == len(fast_r.prompt_logprobs)
                for pos in range(1, len(base_r.prompt_logprobs)):
                    base_lp = base_r.prompt_logprobs[pos]
                    fast_lp = fast_r.prompt_logprobs[pos]
                    assert base_lp is not None and fast_lp is not None
                    prompt_tok_id = base_r.prompt_token_ids[pos]
                    assert prompt_tok_id in base_lp and prompt_tok_id in fast_lp
                    if enforce_eager:
                        assert set(base_lp.keys()) == set(fast_lp.keys())
                        for tok_id, base_entry in base_lp.items():
                            assert fast_lp[tok_id].logprob == pytest.approx(
                                base_entry.logprob, abs=1e-3
                            )
                    else:
                        for tok_id in set(base_lp.keys()) & set(fast_lp.keys()):
                            assert fast_lp[tok_id].logprob == pytest.approx(
                                base_lp[tok_id].logprob, abs=0.75
                            )
        finally:
            fast_llm.llm_engine.engine_core.shutdown()
            del fast_llm
            gc.collect()
            cleanup_dist_env_and_memory()
            torch.accelerator.empty_cache()
