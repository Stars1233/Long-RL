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

import os
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
import numpy as np
from tqdm import tqdm
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
import torch.nn.functional as F
from ...utils.vila_remote_code.constants import IGNORE_INDEX
from ...utils.gpt_api import generate_gpt

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[Dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(multi_modal_data: Dict[str, Any], min_pixels: int, max_pixels: int) -> Dict[str, Any]:
    # may convert image path to image object
    # TODO: add video
    images = []
    for image in multi_modal_data["images"]:
        images.append(process_image(image, min_pixels=min_pixels, max_pixels=max_pixels))

    if len(images) != 0:
        return {"image": images}

    return None

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

OPEN_ENDED_PROMPT = (
    "Compare the following two sentences and determine whether they convey generally similar meanings.\n\n"
    "Respond with only one word: 'Yes' if they are broadly similar or express related ideas (even if wording or details differ), "
    "or 'No' if they are unrelated.\n\n"
    "Sentence 1: '{pred_answer}'\n"
    "Sentence 2: '{ground_truth}'\n\n"
    "Answer:"
)

class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        model_vision_encoder=None,
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True

        if processor is not None and config.limit_images:
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        if "vila" in model_path.lower():
            self.vila_model = True
            model_vision_encoder.config.num_video_frames = config.num_video_frames
            model_vision_encoder.config.fps = 0
            model_path = os.path.join(model_path, "llm")
            self.model_vision_encoder = model_vision_encoder
            self.max_frames_vllm = config.max_frames_vllm if config.max_frames_vllm > 0 else config.num_video_frames
            engine_kwargs["max_model_len"] = config.max_model_len or config.tokens_per_frame * min(
                config.num_video_frames,
                self.max_frames_vllm) + config.prompt_length + config.response_length
            engine_kwargs["disable_mm_preprocessor_cache"] = False
            engine_kwargs["enable_prompt_embeds"]=True
        else:
            self.vila_model = False
            if processor is None: # LLM case
                engine_kwargs["max_model_len"] = config.max_model_len or config.prompt_length + config.response_length
            else: # VLM case
                engine_kwargs["max_model_len"] = config.max_model_len or config.tokens_per_frame * config.num_video_frames + config.prompt_length + config.response_length
            if "omni" in model_path.lower():
                engine_kwargs["max_model_len"] += config.audio_max_length
                engine_kwargs["limit_mm_per_prompt"] = {"audio": 1, "video": 1}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.prompt_length = config.prompt_length
        self.padding_free = config.padding_free
        self.group_frames = config.group_frames
        self.num_chunk_seq = config.num_chunk_seq
        self.bs_vllm = config.bs_vllm
        self.use_cached_embeds = config.use_cached_embeds
        self.tokenizer = tokenizer
        self.open_ended_reward = config.open_ended_reward
        self.format_weight = 0.1

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # TODO: collect input embeds for reuse
        batch_multi_modal_embeds = []
        if batch_multi_modal_data is not None:
            min_pixels, max_pixels = prompts.meta_info["min_pixels"], prompts.meta_info["max_pixels"]
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                if self.vila_model:
                    _raw_prompt_ids = torch.Tensor(list(raw_prompt_ids)).long().unsqueeze(0).to(self.model_vision_encoder.device)
                    vision_key = list(multi_modal_data.keys())[0]
                    _dtype = multi_modal_data[vision_key][0].dtype
                    multi_modal_data[vision_key] = [_data.to(self.model_vision_encoder.dtype) for _data in multi_modal_data[vision_key]]
                    labels = torch.full(_raw_prompt_ids.shape, 1, dtype=_raw_prompt_ids.dtype, device=_raw_prompt_ids.device)
                    media_config = {vision_key: {"frames_split": multi_modal_data[vision_key][0].shape[0] // self.group_frames if "video" in vision_key and self.group_frames>0 else 0}}
                    inputs_embeds, labels, _ = self.model_vision_encoder._embed(_raw_prompt_ids, multi_modal_data, media_config, labels ,None)

                    if not self.use_cached_embeds:
                        resized_embeds = inputs_embeds[labels==IGNORE_INDEX].contiguous()
                        batch_multi_modal_embeds.append(resized_embeds.to(_dtype).cpu())

                    vllm_inputs.append({"prompt_embeds":inputs_embeds.squeeze(0)})
                else:
                    if "audio" in multi_modal_data:
                        vllm_inputs.append(
                            {
                                "prompt_token_ids": list(raw_prompt_ids),
                                "multi_modal_data": multi_modal_data if "video" in multi_modal_data else _process_multi_modal_data(multi_modal_data, min_pixels, max_pixels),
                                "mm_processor_kwargs": {"use_audio_in_video": True,},
                            }
                        )
                    else:
                        vllm_inputs.append(
                            {
                                "prompt_token_ids": list(raw_prompt_ids),
                                "multi_modal_data": multi_modal_data if "video" in multi_modal_data else _process_multi_modal_data(multi_modal_data, min_pixels, max_pixels),
                            }
                        )
            if len(batch_multi_modal_embeds) > 0 and not self.padding_free:
                batch_lengths = [multi_modal_embeds.size(0) for multi_modal_embeds in batch_multi_modal_embeds]
                batch_pad_lengths = [max(batch_lengths) - length for length in batch_lengths]
                for i in range(len(batch_multi_modal_embeds)):
                    if batch_pad_lengths[i] > 0:
                        batch_multi_modal_embeds[i] = F.pad(batch_multi_modal_embeds[i], pad=(0, 0, 0, batch_pad_lengths[i]), value=0.0)
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            if self.bs_vllm > 0:
                completions = []
                for i in tqdm(range(0, len(vllm_inputs), self.bs_vllm)):
                    batch = vllm_inputs[i:i + self.bs_vllm]
                    completions.extend(self.inference_engine.generate(
                        prompts=batch, sampling_params=self.sampling_params, use_tqdm=False
                    ))
            else:
                completions: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False
                )
            response_ids_nopad = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids_nopad, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                ground_truth = _repeat_interleave(non_tensor_batch["ground_truth"], self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)
                if len(batch_multi_modal_embeds) > 0:
                    batch_multi_modal_embeds = _repeat_interleave(batch_multi_modal_embeds, self.sampling_params.n)

            if self.open_ended_reward:
                responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                format_scores = [format_reward(response) for response in responses]

                pred_answers = []

                for response in responses:
                    content_match = re.search(r"<answer>(.*?)</answer>", response)
                    if content_match:
                        pred_answers.append(content_match.group(1).strip())
                    else:
                        pred_answers.append(response.strip())

                judge_prompts = []
                for i in range(len(responses)):
                    judge_prompts.append(OPEN_ENDED_PROMPT.format(pred_answer=pred_answers[i], ground_truth=ground_truth[i]))

                judges = [generate_gpt(prompt) for prompt in judge_prompts]
                accuracy_scores = [1.0 if "yes" in judge.lower() else 0.0 for judge in judges]

                rewards = []
                for i in range(len(judges)):
                    rewards.append({
                        "overall": (1 - self.format_weight) * accuracy_scores[i] + self.format_weight * format_scores[i],
                        "format": format_scores[i],
                        "accuracy": accuracy_scores[i],
                    })


        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
            if len(batch_multi_modal_embeds) > 0:
                non_tensor_batch["multi_modal_embeds"] = batch_multi_modal_embeds
        else:
            non_tensor_batch = {}

        if self.open_ended_reward:
            non_tensor_batch["rewards"] = rewards

        non_tensor_batch["ground_truth"] = ground_truth
        #prompts.meta_info["num_repeat"] = self.sampling_params.n
        prompts.meta_info["num_chunk_seq"] = self.num_chunk_seq
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
