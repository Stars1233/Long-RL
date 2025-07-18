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
"""Utils for tokenization."""
import os

from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModelWithProjection, CLIPTokenizer, UMT5EncoderModel
from verl.utils.diffusion_processor import StableDiffusionProcessor
from verl.utils.wan_processor import WanProcessor
from verl.utils.qwen_omni_utils import Qwen2_5OmniProcessor


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    if kwargs.get("diffusion", False):
        return get_diffusion_tokenizer(model_path, **kwargs)

    """Create a huggingface pretrained tokenizer."""
    if "vila" in model_path.lower():
        model_path = os.path.join(model_path, "llm")
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        tokenizer.chat_template = override_chat_template

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_diffusion_tokenizer(model_path: str, **kwargs) -> PreTrainedTokenizer:
    if "wan" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", **kwargs)
        return tokenizer
    elif 'stable-diffusion' in model_path.lower():
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer_2",
        )
        tokenizer_3 = T5TokenizerFast.from_pretrained(
            model_path,
            subfolder="tokenizer_3",
        )

        tokenizers = [tokenizer_1, tokenizer_2, tokenizer_3]
        return tokenizers
    else:
        raise ValueError(f"Unsupported model: {model_path}")


def get_processor(model_path: str, num_video_frames: int = 8, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    if kwargs.get("diffusion", False):
        return get_diffusion_processor(model_path, **kwargs)

    if "vila" in model_path.lower():
        kwargs["trust_remote_code"] = True

    if "omni" in model_path.lower():
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, **kwargs)
    else:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if hasattr(processor, "config"):
        processor.config.num_video_frames = num_video_frames
        processor.config.fps = 2
    if override_chat_template is not None:
        processor.chat_template = override_chat_template

    processor.num_video_frames = num_video_frames
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor


def get_diffusion_processor(model_path: str, **kwargs) -> Optional[ProcessorMixin]:
    kwargs_clean = {k: v for k, v in kwargs.items() if k != "diffusion"}

    if "wan" in model_path.lower():
        text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer", **kwargs)
        text_encoder.requires_grad_(False)

        processor = WanProcessor(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_sequence_length=kwargs.get('max_sequence_length', 128),
        )
        return processor
    elif 'stable-diffusion' in model_path.lower():
        text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder="text_encoder",
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
        )
        text_encoder_3 = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_3",
        )

        # 冻结参数以节省显存
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        text_encoder_3.requires_grad_(False)

        # 直接加载三个tokenizer
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer_2",
        )
        tokenizer_3 = T5TokenizerFast.from_pretrained(
            model_path,
            subfolder="tokenizer_3",
        )

        text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]
        tokenizers = [tokenizer_1, tokenizer_2, tokenizer_3]

        processor = StableDiffusionProcessor(
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            max_sequence_length=kwargs.get('max_sequence_length', 128),
        )

        return processor