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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from .flops_counter import VALID_MODLE_TYPE
from ..models.transformers.qwen2_vl import get_rope_index
from ..models.transformers.qwen2_5_omni import get_rope_index_omni
from . import torch_functional as VF
import re
from functools import partial
from verl.utils.vila_remote_code.tokenizer_utils import tokenize_conversation
from verl.utils.vila_remote_code.auto_processor import extract_value_from_conv
from verl.utils.qwen_vl_utils import process_vision_info
from verl.utils.qwen_omni_utils import process_mm_info
import torchvision.transforms.functional as TF
from verl.utils.diffusion_processor import StableDiffusionProcessor
from verl.utils.wan_processor import WanProcessor


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given image, and then provide the final number. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"
QUESTION_TEMPLATE_VIDEO = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"
QUESTION_TEMPLATE_VIDEO_QWEN = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> the letter of your choice (A, B, C, or D) </answer>.\n\n Question: {question}"
QUESTION_TEMPLATE_OMNI = "Please first think deeply about the question based on the given video and audio, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> the letter of your choice (A, B, C, or D) </answer>.\n\n Question: {question}"


def _get_messages_vila(example: Dict[str, Any],
                       prompt_key: str = "prompt",
                       image_key: str = "images",
                       image_dir: Optional[str] = None,
                       video_key: str = "videos",
                       video_dir: str = None,) -> Dict[str, Any]:
    if video_key in example:
        vision_key = "video"
        vision_value = example[video_key]
        if video_dir is not None and isinstance(vision_value, str):  # image paths
            vision_value = os.path.join(video_dir, vision_value)
        message_key = "video"
        question_template = QUESTION_TEMPLATE_VIDEO
    elif image_key in example:
        vision_key = "image"
        vision_value = example[image_key][0]
        if isinstance(vision_value, ImageObject):
            message_key = "image_pil"
        elif isinstance(vision_value, str):
            vision_key = "image"
        else:
            raise ValueError("Unknown image type", vision_value)
        question_template = QUESTION_TEMPLATE_IMAGE
    else:
        raise ValueError("Unsupported VILA for text only.")

    messages = [{"role": "user", "content": "<%s>" % vision_key + example[prompt_key]}]
    prompt = question_template.format(question=messages[-1]['content'].replace("<%s>" % vision_key, ""))
    messages[-1]['content'] = [
        {"type": vision_key, message_key: vision_value},
        {"type": "text", "text": prompt},
    ]
    return messages, prompt

def _filter_overlong_prompts_vila(example: Dict[str, Any],
                                  tokenizer=None,
                                  max_prompt_length: int = 2048,
                                  prompt_key: str = "prompt",
                                  image_key: str = "images",
                                  image_dir: Optional[str] = None,
                                  video_key: str = "videos",
                                  video_dir: str = None,) -> bool:
    def apply_chat_template_vila(conversation):
        vila_conv = []
        for chat in conversation:
            vila_chat = {"from": "", "value": []}
            if chat["role"] in ("user", "system"):
                # user allows to input image and text
                vila_chat["from"] = "human" if chat["role"] == "user" else "system"
                vila_chat["value"] = extract_value_from_conv(chat)
            elif chat["role"] == "assistant":
                vila_chat["from"] = "gpt"
                vila_chat["value"] = extract_value_from_conv(chat)
            else:
                raise ValueError(f"Unsupported role: {chat['role']} in chat {chat}")
            vila_conv.append(vila_chat)
        return vila_conv

    messages, _ = _get_messages_vila(example, prompt_key, image_key, image_dir, video_key, video_dir)
    messages = apply_chat_template_vila(messages)
    messages[-1]['value'] = messages[-1]['value'][-1]
    inputs = tokenize_conversation(messages, tokenizer, add_generation_prompt=True,
                                   return_ids_only=False)
    return inputs.input_ids[0].size(-1) <= max_prompt_length

class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        image_dir: Optional[str] = None,
        video_key: str = "videos",
        video_dir: str = None,
        cache_dir: str = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        vila_model: bool = False,
        diffusion: bool = False,
        is_omni: bool = False,
        audio_max_length: int = 10000,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.image_dir = image_dir
        self.video_key = video_key
        self.video_dir = video_dir
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.filter_overlong_prompts = filter_overlong_prompts
        self.vila_model = vila_model
        self.cache_dir = cache_dir
        self.is_omni = is_omni
        self.audio_max_length = audio_max_length

        self.video_hw = {}
        self.num_tokens_per_frame = -1
        self.video_backup = None
        self.diffusion = diffusion

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.filter_overlong_prompts:
            if self.vila_model:
                _filter_overlong_prompts = partial(_filter_overlong_prompts_vila, tokenizer=self.tokenizer,
                                                   max_prompt_length=max_prompt_length, prompt_key=prompt_key,
                                                   image_key=image_key, image_dir=image_dir, video_key=video_key,
                                                   video_dir=video_dir)
            else:
                _filter_overlong_prompts = self._filter_overlong_prompts
            self.dataset = self.dataset.filter(
                _filter_overlong_prompts, desc="Filtering overlong prompts", num_proc=16,
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.is_omni:
            system_omni = {
                "role": "system",
                "content": [
                    {"type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            }
            prompt_str = QUESTION_TEMPLATE_OMNI.format(question=example[self.prompt_key])
            video = example[self.video_key]
            if self.video_dir is not None:  # video paths
                video = os.path.join(self.video_dir, video)

            messages = [system_omni, {"role": "user", "content": [
                {"type": "video", "video": video, "nframes": self.processor.num_video_frames},
                {"type": "text", "text": prompt_str}]}]

            if "resized_height" in self.video_hw and "resized_width" in self.video_hw:
                messages[1]["content"][0]["resized_height"] = self.video_hw["resized_height"]
                messages[1]["content"][0]["resized_width"] = self.video_hw["resized_width"]
            return messages


        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            video = example[self.video_key]
            assert isinstance(video, str), "Only support video path input"
            if self.video_dir is not None:  # video paths
                video = os.path.join(self.video_dir, video)
            prompt_str = QUESTION_TEMPLATE_VIDEO_QWEN.format(question=example[self.prompt_key])
            content = [{"type": "video", "video": video, "nframes": self.processor.num_video_frames}, {"type": "text", "text": prompt_str}]
            if "resized_height" in self.video_hw and "resized_width" in self.video_hw:
                content[0]["resized_height"] = self.video_hw["resized_height"]
                content[0]["resized_width"] = self.video_hw["resized_width"]
            return [{"role": "user", "content": content}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key] or []
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            resized_images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for image in images
            ] or None
            model_inputs = self.processor(resized_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        max_prompt_length = self.max_prompt_length
        if self.vila_model:
            messages, prompt = _get_messages_vila(example, self.prompt_key, self.image_key, self.image_dir, self.video_key, self.video_dir)
            vision_key = messages[-1]['content'][0]['type']
            videos_cache = []
            num_video_frames = self.processor.config.num_video_frames
            if self.cache_dir is not None:
                cache_path = os.path.join(self.cache_dir, example[self.video_key].split(".")[0]+".pt")
                if os.path.exists(cache_path):
                    videos_cache.append(torch.load(cache_path, map_location=torch.device('cpu')))
                    self.processor.config.num_video_frames = 1

            messages = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.processor(text=[messages], return_tensors="pt")
            example["multi_modal_data"] = {
                vision_key: videos_cache if len(videos_cache) else model_inputs['media'][vision_key],
            }
            self.processor.config.num_video_frames = num_video_frames
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["raw_prompt_ids"] = input_ids.tolist()
        elif self.diffusion:
            assert self.prompt_key in example, f"{self.prompt_key} is required for diffusion"
            # print("*** example ***", example)
            if isinstance(self.processor, WanProcessor):
                prompt_embeds = self.processor(example[self.prompt_key])
                negative_prompt_embeds = self.processor("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
            elif isinstance(self.processor, StableDiffusionProcessor):
                prompt_embeds, pooled_prompt_embeds = self.processor(example[self.prompt_key])
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.processor("")
                example["pooled_prompt_embeds"] = pooled_prompt_embeds
                example["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            else:
                raise ValueError(f"Processor {self.processor} is not supported.")

            # print("*** dataset prompt_embeds ***", prompt_embeds.shape)
            # print("*** dataset pooled_prompt_embeds ***", pooled_prompt_embeds.shape)
            example["prompt_embeds"] = prompt_embeds
            example["negative_prompt_embeds"] = negative_prompt_embeds

            return example
        else:
            if self.image_key in example:
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = example.pop(self.image_key)
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                    images = [os.path.join(self.image_dir, image) for image in images]
                resized_images = [
                    process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for image in images
                ] or None
                model_inputs = self.processor(resized_images, [prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                example["multi_modal_data"] = {"images": images}
                #image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                #max_prompt_length += (input_ids==image_token_id).sum()
            elif self.video_key in example:
                if not self.is_omni:
                    prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    videos = None
                    if self.cache_dir is not None:
                        cache_path = os.path.join(self.cache_dir, example[self.video_key].split(".")[0]+".pt")
                        if os.path.exists(cache_path):
                            video_cache = torch.load(cache_path).float()
                            video_cache_frames = video_cache.size(0)
                            if self.processor.num_video_frames != video_cache_frames:
                                print("Disable using cache video, because the requires num of video frames %d should equal to the cached frames %d" % (self.processor.num_video_frames, video_cache_frames))
                                self.cache_dir = None
                            else:
                                videos = [video_cache]

                    if videos is None:
                        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

                    if len(videos) > 0 and not "resized_height" in self.video_hw and not "resized_width" in self.video_hw :
                        self.video_hw["resized_height"], self.video_hw["resized_width"] = videos[0].size(2), videos[0].size(3)
                        self.video_backup = videos

                    if self.video_backup:
                        if videos[0].size() != self.video_backup[0].size():
                            print("Video size mismatch, use backup video instead.")
                            videos = self.video_backup

                    model_inputs = self.processor(text=prompt, images=None, videos=videos, padding=True, return_tensors="pt")
                    input_ids = model_inputs.pop("input_ids")[0]
                    attention_mask = model_inputs.pop("attention_mask")[0]
                    example["multi_modal_data"] = {"video": videos}

                    if self.num_tokens_per_frame < 0:
                        video_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
                        total_video_tokens = (input_ids==video_token_id).sum()
                        self.num_tokens_per_frame = total_video_tokens // self.processor.num_video_frames
                    max_prompt_length += (self.num_tokens_per_frame * self.processor.num_video_frames)
                else:
                    prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[0]
                    audios, images, videos, video_kwargs = process_mm_info(messages, use_audio_in_video=True,
                                                                           return_video_kwargs=True)
                    if len(videos) > 0 and not "resized_height" in self.video_hw and not "resized_width" in self.video_hw:
                        self.video_hw["resized_height"], self.video_hw["resized_width"] = videos[0].size(2), videos[
                            0].size(3)
                    model_inputs = self.processor(text=[prompt], audio=audios, images=images, videos=videos, padding=True,
                                                  return_tensors="pt", use_audio_in_video=True, **video_kwargs)
                    input_ids = model_inputs.pop("input_ids")[0]
                    attention_mask = model_inputs.pop("attention_mask")[0]
                    example["multi_modal_data"] = {"video": videos, "audio": audios}
                    feature_attention_mask = model_inputs.pop("feature_attention_mask")
                    audio_feature_length = torch.sum(feature_attention_mask, dim=1)
                    audio_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|VIDEO|>")
                    max_prompt_length += (input_ids == audio_token_id).sum()
                    max_prompt_length += self.audio_max_length  # audio max_length
            else:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.max_prompt_length:
                if self.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
                elif self.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                elif self.truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
            example["raw_prompt_ids"] = raw_prompt_ids

        if self.is_omni:
            position_ids, mrope_position_deltas = get_rope_index_omni(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask,
                use_audio_in_video=True,
                audio_seqlens=audio_feature_length,
            )
        elif self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=self.vila_model or self.diffusion, #True,
            truncation=self.truncation,
        )

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        answer = example.pop(self.answer_key)
        if "<answer>" in answer:
            match = re.search(r"<answer>(.*?)</answer>", answer)
            example["ground_truth"] = match.group(1)
        else:
            example["ground_truth"] = answer
        return example
