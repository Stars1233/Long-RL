#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import html
import torch
import regex as re
from typing import Union, List, Optional
from transformers import ProcessorMixin, AutoTokenizer, UMT5EncoderModel
from transformers.utils import is_ftfy_available

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    """Basic text cleaning using ftfy."""
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Clean up whitespace in text."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    """Clean prompt text by applying basic and whitespace cleaning."""
    text = whitespace_clean(basic_clean(text))
    return text


class WanProcessor(ProcessorMixin):
    """
    Wan model processor that encodes prompts using UMT5 text encoder.
    
    This processor handles text encoding for Wan video generation models
    using UMT5EncoderModel and AutoTokenizer.
    """
    
    attributes = ["text_encoder", "tokenizer"]
    
    def __init__(self, text_encoder: UMT5EncoderModel, tokenizer: AutoTokenizer, max_sequence_length: int = 226):
        """
        Initialize the WanProcessor.
        
        Args:
            text_encoder: UMT5 text encoder model
            tokenizer: Corresponding tokenizer
            max_sequence_length: Maximum sequence length for encoding
        """
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.text_encoder.requires_grad_(False)
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Encode prompt using T5 encoder.
        
        Args:
            prompt: Text prompt(s) to encode
            num_videos_per_prompt: Number of videos to generate per prompt
            max_sequence_length: Maximum sequence length (uses self.max_sequence_length if None)
            device: Target device
            dtype: Target dtype
            
        Returns:
            Encoded prompt embeddings
        """
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length
            
        device = device or (self.text_encoder.device if hasattr(self.text_encoder, 'device') else torch.device('cpu'))
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            # return_attention_mask=True,
            return_tensors="pt",
        )
        # text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        text_input_ids = text_inputs.input_ids

        # seq_lens = mask.gt(0).sum(dim=1).long()

        # prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        # prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        # prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        # prompt_embeds = torch.stack(
        #     [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        # )
        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Encode prompts into text encoder hidden states.

        Args:
            prompt: Prompt(s) to be encoded
            negative_prompt: Negative prompt(s) for classifier-free guidance
            do_classifier_free_guidance: Whether to use classifier-free guidance
            num_videos_per_prompt: Number of videos to generate per prompt
            prompt_embeds: Pre-generated text embeddings
            negative_prompt_embeds: Pre-generated negative text embeddings
            max_sequence_length: Maximum sequence length
            device: Target device
            dtype: Target dtype
            
        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds)
        """
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length
            
        device = device or (self.text_encoder.device if hasattr(self.text_encoder, 'device') else torch.device('cpu'))

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds.to(torch.bfloat16)
    
    def __call__(self, prompt: Union[str, List[str]], **kwargs):
        """
        Forward method that calls encode_prompt.
        
        Args:
            prompt: Text prompt(s) to encode
            **kwargs: Additional arguments passed to encode_prompt
            
        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds)
        """
        return self.encode_prompt(prompt, **kwargs)