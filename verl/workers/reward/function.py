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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[List[RewardInput]], List[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer
        self.diffusion = config.diffusion

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            if "rewards" in data.non_tensor_batch:
                score = data.non_tensor_batch["rewards"][i]
            else:
                valid_response_ids = response_ids[i][: response_length[i]]
                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
                )
                score = self.reward_fn(
                    {
                        "response": response_str,
                        "response_length": response_length[i],
                        "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    }
                )
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        if self.diffusion:
            # For diffusion, return a single value tensor
            reward_inputs = []
            for i in range(len(data)):
                if "images" in data.batch:
                    images = data.batch["images"][i]
                    reward_inputs.append(
                    {
                        "images": images,
                    }
                    )
                elif "videos" in data.batch:
                    videos = data.batch["videos"][i]
                    reward_inputs.append(
                    {
                        "videos": videos,
                    }
                    )

            reward_tensor = torch.zeros((len(data), 1), dtype=torch.float32)
            reward_metrics = defaultdict(list)
            scores = self.reward_fn(reward_inputs)
            # print("***scores***", scores)
            for i, score in enumerate(scores):
                reward_tensor[i, 0] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
            return reward_tensor, reward_metrics

        else:
            reward_inputs = []
            response_ids = data.batch["responses"]
            response_length = data.batch["response_mask"].sum(dim=-1)
            for i in range(len(data)):
                valid_response_ids = response_ids[i][: response_length[i]]
                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
                )
                reward_inputs.append(
                    {
                        "response": response_str,
                        "response_length": response_length[i],
                        "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    }
                )

            scores = self.reward_fn(reward_inputs)
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            for i, score in enumerate(scores):
                reward_tensor[i, response_length[i] - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

            return reward_tensor, reward_metrics
