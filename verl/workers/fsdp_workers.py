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
The main entry point to run the PPO algorithm
"""

from typing import Literal, Optional, Union, cast
import os
import numpy as np
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
    AutoModel,
)
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig, Qwen2_5OmniThinkerConfig)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniPreTrainedModel, Qwen2_5OmniPreTrainedModelForConditionalGeneration,Qwen2_5OmniThinkerForConditionalGeneration
from transformers.modeling_utils import no_init_weights
from diffusers import StableDiffusion3Pipeline, WanPipeline

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout import vLLMRollout, StableDiffusionRollout, WanRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_ref_param_offload = self.config.ref.offload.offload_params

        if self.config.vila_model:
            self.config.actor.vila_model = True
            self.config.ref.vila_model = True

        self.diffusion = self.config.actor.diffusion

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        # create main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        # create ulysses device mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # validate and normalize config
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = (
            config.global_batch_size * config.ulysses_size
        ) // self.device_mesh.size()
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        ulysses_size: int,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        if self.diffusion:
            self._build_model_optimizer_diffusion(model_config, fsdp_config, optim_config, role)
            return

        if role != "ref":  # ref model's tokenizer is same as actor
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = AutoConfig.from_pretrained(
                os.path.join(model_config.model_path, "llm") if self.config.vila_model else model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )

            if hasattr(self.model_config, "num_attention_heads"):
                assert self.model_config.num_attention_heads % ulysses_size == 0, " num_attention_heads %d must be divisible by ulysses_size %d."%(self.model_config.num_attention_heads, ulysses_size)

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        if ulysses_size > 1:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if role == "critic":
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
        else:
            auto_class = AutoModelForCausalLM

        if self.config.is_omni:
            auto_class = Qwen2_5OmniThinkerForConditionalGeneration
            auto_class._tp_plan = []
            self.model_config = Qwen2_5OmniThinkerConfig.from_pretrained(
                    model_config.model_path,
                    trust_remote_code=model_config.trust_remote_code,
                    bos_token_id="null",
                    eos_token_id=151645,
                    pad_token_id=151643,
                    **model_config.override_config,
                )
            self.generation_config.bos_token_id = "null"
            self.generation_config.eos_token_id = 151645
            self.generation_config.pad_token_id = 151643

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = auto_class.from_pretrained(
                os.path.join(model_config.model_path, "llm") if self.config.vila_model else model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                if self.config.is_omni:
                    self.model_config.torch_dtype = torch_dtype
                    model = auto_class._from_config(
                        self.model_config,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                else:
                    model = auto_class.from_config(
                            self.model_config,
                            torch_dtype=torch_dtype,
                            attn_implementation="flash_attention_2",
                            trust_remote_code=model_config.trust_remote_code,
                        )

        model = cast(PreTrainedModel, model)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if role == "ref":
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):  # transformers >= 4.52.0
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):  # transformers < 4.52.0
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if role in ["actor", "critic"]:
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_model_optimizer_diffusion(self, model_config: ModelConfig, fsdp_config: FSDPConfig, optim_config: Optional[OptimConfig], role: Literal["actor", "critic", "ref"]):
        if role != "ref":  # ref model's tokenizer is same as actor
            self.processor = get_processor(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
                diffusion=True,
            )
            if "wan" in model_config.model_path.lower():
                self.model_config = WanPipeline.load_config(model_config.model_path)
            else:
                self.model_config = StableDiffusion3Pipeline.load_config(model_config.model_path)
            self.print_rank0(f"Model config: {self.model_config}")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            if "wan" in model_config.model_path.lower():
                model = WanPipeline.from_pretrained(
                    model_config.model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                ).transformer
            else:
                model = StableDiffusion3Pipeline.from_pretrained(
                    model_config.model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                ).transformer
            if fsdp_config.enable_rank0_init:
                model.to("cpu")
        else:
            with no_init_weights(), init_empty_weights():
                if "wan" in model_config.model_path.lower():
                    model = WanPipeline.from_pretrained(
                        model_config.model_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                    ).transformer
                else:
                    model = StableDiffusion3Pipeline.from_pretrained(
                        model_config.model_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                    ).transformer
            if fsdp_config.enable_rank0_init:
                model.to("cpu")

        model = cast(PreTrainedModel, model)  # lint
        # model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)

        if role == "ref":
            model.requires_grad_(False)

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if role in ["actor", "critic"]:
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_rollout(self) -> None:
        if self.diffusion:
            self._build_rollout_diffusion()
            return

        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model_vision_encoder=self.model_vision_encoder,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
            vila_model="vila" in self.config.actor.model.model_path.lower(),
        )
        print_gpu_memory_usage("After vllm init")

    def _build_rollout_diffusion(self) -> None:
        if "wan" in self.config.actor.model.model_path.lower():
            self.rollout = WanRollout(
                model_path=self.config.actor.model.model_path,
                config=self.config.rollout,
            )
        elif "stable-diffusion" in self.config.actor.model.model_path.lower():
            self.rollout = StableDiffusionRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
        )
        else:
            raise ValueError(f"Model {self.config.actor.model.model_path} is not supported.")
        print_gpu_memory_usage("After Rollout init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                ulysses_size=self.config.critic.ulysses_size,
                role="critic",
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                ulysses_size=self.config.actor.ulysses_size,
                role="actor",
            )

        if self._has_ref:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                ulysses_size=self.config.ref.ulysses_size,
                role="ref",
            )

        self.model_vision_encoder = None
        if self._has_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import
            if self.config.vila_model:
                self.model_vision_encoder = AutoModel.from_pretrained(self.config.actor.model.model_path,
                                                                      trust_remote_code=True,
                                                                      torch_dtype=torch.bfloat16,
                                                                      device_map="auto",
                                                                      llm_only_need_embed=True)
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
                model_vision_encoder=self.model_vision_encoder,
            )

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:  # must after actor
            self._build_rollout()

        if self._has_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_fsdp_module,
                model_vision_encoder=self.model_vision_encoder,
            )

        if self._has_actor or self._has_critic:
            if not self.diffusion:
                self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=None if self.diffusion else self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path, save_model_only, self.diffusion)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        if "uid" in self._cache and not np.all(data.non_tensor_batch["uid"] == self._cache["uid"]):
            self._cache.clear()

        if not self.config.vila_model:
            if "multi_modal_inputs" not in self._cache:
                min_pixels = data.meta_info["min_pixels"]
                max_pixels = data.meta_info["max_pixels"]
                batch_multi_modal_inputs = []
                num_repeat = 1
                for i, multi_modal_data in enumerate(data.non_tensor_batch["multi_modal_data"]):
                    if "images" in multi_modal_data:
                        images = []
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels=min_pixels, max_pixels=max_pixels))

                        if len(images) != 0:
                            # it's necessary to add `dict` to properly convert batch features to dict
                            # otherwise the batch features will be converted to dict keys
                            # see https://github.com/hiyouga/EasyR1/pull/339
                            multi_modal_inputs = dict(self.processor.image_processor(images=images, videos=None, return_tensors="pt"))
                            multi_modal_inputs = {k: v.to(torch.cuda.current_device()) for k, v in multi_modal_inputs.items()}
                            batch_multi_modal_inputs.append(multi_modal_inputs)
                        else:
                            batch_multi_modal_inputs.append({})
                    elif "video" in multi_modal_data:
                        videos = multi_modal_data["video"]
                        if len(videos) != 0:
                            # it's necessary to add `dict` to properly convert batch features to dict
                            # otherwise the batch features will be converted to dict keys
                            # see https://github.com/hiyouga/EasyR1/pull/339
                            if "num_repeat" in data.meta_info:
                                num_repeat = data.meta_info["num_repeat"]

                            if i % num_repeat > 0:
                                continue
                            multi_modal_inputs = dict(self.processor.video_processor(images=None, videos=[video.to(torch.cuda.current_device()) for video in videos], return_tensors="pt"))
                            multi_modal_inputs = {k: v.to(torch.cuda.current_device()) for k, v in multi_modal_inputs.items()}
                            for k in multi_modal_inputs:
                                if "pixel_values" in k:
                                    multi_modal_inputs[k] = multi_modal_inputs[k].to(torch.bfloat16)
                            batch_multi_modal_inputs.extend([multi_modal_inputs] * num_repeat)
                        else:
                            batch_multi_modal_inputs.append({})

                self._cache["uid"] = data.non_tensor_batch["uid"]
                self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

            data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]
        del data.non_tensor_batch["multi_modal_data"]

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            if not self.diffusion:
                global_num_tokens = data.meta_info["global_token_num"]
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
                metrics["perf/mfu_actor"] = (
                    estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
                )
                metrics["perf/max_memory_allocated_gb"] = (
                    torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
                ) / (1024**3)
                metrics["perf/max_memory_reserved_gb"] = (
                    torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
                ) / (1024**3)
                metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        return output.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout
        if self.diffusion:
            output = self.rollout.generate_sequences(prompts=prompts)
        else:
            meta_info = {
                "eos_token_id": self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id,
            }
            prompts.meta_info.update(meta_info)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.rollout_sharding_manager.postprocess_data(output)

        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            if self.diffusion:
                log_probs, prev_sample_mean = self.actor.compute_log_prob(data=data)
                output = DataProto.from_dict(tensors={"old_log_probs": log_probs, "old_prev_sample_mean": prev_sample_mean})
            else:
                output = self.actor.compute_log_prob(data=data)
                output = DataProto.from_dict(
                    tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
                )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_ref_param_offload:
            load_fsdp_model(self.ref_fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            if self.diffusion:
                log_probs, prev_sample_mean = self.ref_policy.compute_log_prob(data=data)
                output = DataProto.from_dict(tensors={"ref_log_probs": log_probs, "ref_prev_sample_mean": prev_sample_mean})
            else:
                output = self.ref_policy.compute_log_prob(data=data)
                output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_fsdp_module._handle.reshard(True)

        if self._use_ref_param_offload:
            offload_fsdp_model(self.ref_fsdp_module)

        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    metric: np.array([value] if np.isscalar(value) else value) for metric, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        return output.to("cpu")
