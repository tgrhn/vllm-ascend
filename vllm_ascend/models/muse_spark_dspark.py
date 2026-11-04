# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's DSpark draft, which Ascend does not run yet.

Upstream registers ``MuseSparkDSparkModel`` and builds its layers from the
vendor Muse Spark stack, which reaches CUDA-only code on Ascend: the attention
rotates through ``RotaryEmbedding.forward_cuda``, every RMSNorm calls a
CUDA-tuned Triton kernel, and the output gate runs the FLA Triton kernel. The
Ascend Muse Spark adapter replaces each of these for the target model only.
Registering this class in the draft's place stops speculative decoding with
the draft at load time with a clear error, instead of inside a CUDA op on the
first draft step.

Lives apart from ``muse_spark`` so that serving the target alone never imports
the upstream draft module.
"""

from vllm.config import VllmConfig
from vllm.models.muse_spark.common.dspark import MuseSparkDSparkForCausalLM


class AscendMuseSparkDSparkForCausalLM(MuseSparkDSparkForCausalLM):
    """The upstream draft class, refused at construction.

    Subclassing keeps the upstream interfaces, so the registry inspects this
    class as the draft it replaces and the error below is the one raised.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        raise NotImplementedError(
            "The Muse Spark DSpark draft is not supported on Ascend yet: its layers run "
            "upstream's CUDA/ROCm Muse Spark kernels. Serve Muse Spark without speculative decoding."
        )
