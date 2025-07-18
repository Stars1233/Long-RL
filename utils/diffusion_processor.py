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

import torch
from transformers import ProcessorMixin


class StableDiffusionProcessor(ProcessorMixin):
    """
    Diffusion model processor that encodes prompts using multiple text encoders.
    
    This processor combines CLIP and T5 text encoders to create embeddings
    suitable for diffusion models.
    """
    
    attributes = ["text_encoders", "tokenizers"]
    
    def __init__(self, text_encoders, tokenizers, max_sequence_length=256):
        """
        Initialize the DiffusionProcessor.
        
        Args:
            text_encoders: List of text encoders (CLIP encoders + T5 encoder)
            tokenizers: List of corresponding tokenizers
            max_sequence_length: Maximum sequence length for T5 encoder
        """
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.max_sequence_length = max_sequence_length
    
    def _encode_prompt_with_t5(
        self,
        text_encoder,
        tokenizer,
        max_sequence_length,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _encode_prompt_with_clip(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None,
    ):
        """
        Encode prompts using both CLIP and T5 text encoders.
        
        Args:
            prompt: Text prompt to encode
            device: Device to use for computation
            num_images_per_prompt: Number of images to generate per prompt
            text_input_ids_list: Pre-tokenized input IDs (optional)
            
        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_tokenizers = self.tokenizers[:2]
        clip_text_encoders = self.text_encoders[:2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device if device is not None else text_encoder.device,
                num_images_per_prompt=num_images_per_prompt,
                text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = self._encode_prompt_with_t5(
            self.text_encoders[-1],
            self.tokenizers[-1],
            self.max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
            device=device if device is not None else self.text_encoders[-1].device,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds.to(torch.bfloat16), pooled_prompt_embeds.to(torch.bfloat16)
    
    def __call__(self, prompt: str, **kwargs):
        """
        Forward method that calls encode_prompt.
        
        Args:
            prompt: Text prompt to encode
            **kwargs: Additional arguments passed to encode_prompt
            
        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        # print("*** prompt ***", prompt)
        # print("*** self.text_encoders[0].device ***", self.text_encoders[0].device)
        # print("*** self.text_encoders[1].device ***", self.text_encoders[1].device)
        # print("*** self.text_encoders[2].device ***", self.text_encoders[2].device)
        return self.encode_prompt(prompt, **kwargs)