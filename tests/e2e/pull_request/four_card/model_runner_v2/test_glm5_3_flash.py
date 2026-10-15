# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash functional smoke on model runner V2, not accuracy or performance.

The V1 sibling (``four_card/test_glm5_3_flash.py``) pins
``VLLM_USE_V2_MODEL_RUNNER=0``, so the Flash path had no V2 coverage while the
open V2 work for this model lands. This file reuses that sibling's synthetic
model, dummy loader and worker checks and runs them on V2 instead, so the two
runners stay in step.

Eager only: that is the configuration GLM-5.3-Flash is validated for on V2
today. Prefix caching stays off for the same reason, matching the sibling.
"""

import os
from unittest.mock import patch

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

# Importing the sibling also registers its dummy model loader, which this
# test depends on. ``FlashWorker`` is resolved by name in the worker
# subprocess, so it must be addressed through the module that defines it.
from tests.e2e.pull_request.four_card import test_glm5_3_flash as flash_v1

V1_MODULE = flash_v1.__name__


def _settings(model_path):
    """Mirror of the V1 sibling's settings, V2 and eager.

    Kept local rather than imported so the V2 case can diverge later without
    editing the V1 test.
    """
    return dict(
        load_format=flash_v1.LOADER,
        worker_extension_cls=f"{V1_MODULE}.FlashWorker",
        dtype="bfloat16",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        quantization="ascend",
        distributed_executor_backend="mp",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        limit_mm_per_prompt={"image": 0, "video": 0},
        block_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        seed=1024,
        enforce_eager=True,
        additional_config={"enable_cpu_binding": False, "enable_fused_mc2": 0},
    )


@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "400",
    },
)
def test_glm53flash_tp4_v2(tmp_path):
    model = tmp_path / "model"
    config = flash_v1._write_model(model)
    assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
    with VllmRunner(str(model), **_settings(model)) as runner:
        assert runner.model.collective_rpc("check_flash_paths") == [True] * 4
        params = SamplingParams(
            temperature=0,
            max_tokens=flash_v1.OUTPUT_TOKENS,
            ignore_eos=True,
            detokenize=False,
        )
        # A single request followed by a mixed batch exercises both API shapes.
        for lengths in ((128,), (127, 128, 129)):
            outputs = runner.model.generate(
                [{"prompt_token_ids": [10 + i % 1000 for i in range(length)]} for length in lengths],
                params,
                use_tqdm=False,
            )
            assert len(outputs) == len(lengths)
            for output in outputs:
                assert output.finished and len(output.outputs) == 1
                tokens = output.outputs[0].token_ids
                assert len(tokens) == flash_v1.OUTPUT_TOKENS
                assert all(0 <= token < config["text_config"]["vocab_size"] for token in tokens)
