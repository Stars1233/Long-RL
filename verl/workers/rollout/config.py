# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    val_override_config: Dict[str, Any] = field(default_factory=dict)
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)
    num_video_frames: int = field(default=8, init=False)
    tokens_per_frame: int = field(default=256, init=False)
    padding_free: bool = field(default=False, init=False)
    group_frames: int = 64
    max_frames_vllm: int = -1
    diffusion: bool = False
    guidance_scale: float = 4.5
    resolution: int = 768
    num_steps: int = 10
    kl_reward: float = 0.0
    num_frames: int = 81
    height: int = 480
    width: int = 832
    audio_max_length: int = 10000  # max length of audio feature, used for padding in dataset
    num_chunk_seq: int = -1
    bs_vllm: int = -1
    use_cached_embeds: bool = False
    open_ended_reward: bool = False

    def to_dict(self):
        return asdict(self)
