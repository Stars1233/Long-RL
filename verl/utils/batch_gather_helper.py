# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import pickle
import torch
import torch.distributed as dist
from copy import deepcopy


def check_object_size(obj, name="object", limit_mb=500):
    b = pickle.dumps(obj)
    size_mb = len(b) / (1024 * 1024)
    print(f"[BatchGatherHelper] Size of `{name}` ~ {size_mb:.2f} MB")
    if size_mb > limit_mb:
        print(f"⚠️  `{name}` exceeds {limit_mb} MB! Consider chunking or tensor gather instead.")
    return size_mb


def chunk_dict_list(input_list, num_chunk_seq):
    new_list = []
    for d in input_list:
        video = d["pixel_values_videos"]  # shape [N, dim]
        N = video.shape[0]
        assert N % num_chunk_seq == 0, f"N={N} must be divisible by num_chunk_seq={num_chunk_seq}"

        chunks = torch.chunk(video, num_chunk_seq, dim=0)

        for chunk in chunks:
            new_d = dict(d)
            new_d["pixel_values_videos"] = chunk
            new_list.append(new_d)
    return new_list


def merge_chunked_list(chunks):
    _chunk = chunks[0]
    for i in range(len(_chunk)):
        _chunk[i]["pixel_values_videos"] = torch.cat([d[i]["pixel_values_videos"] for d in chunks], dim=0)
    return merged_list


def smart_gather_object(
    obj,
    size,
    group=None,
    limit_mb=500,
    num_repeat=1,
    num_chunk_seq=1
):
    key = None
    for k in ["data", "inputs", "embeds"]:
        if "multi_modal_%s" %k in obj:
            key = "multi_modal_%s" %k

    obj_size = check_object_size(obj, name="non_tensor_batch", limit_mb=limit_mb)

    if obj_size < limit_mb or key is None:
        print(f"[BatchGatherHelper] Using standard all_gather_object.")
        output = [None for _ in range(size)]
        dist.all_gather_object(output, obj, group=group)
        return output

    print(f"[BatchGatherHelper] Using selective chunked gather for {key}.")

    multi_modal_data = obj[key]
    obj_copy = {k: v for k, v in obj.items() if k != key}

    base_output = [None for _ in range(size)]
    dist.all_gather_object(base_output, obj_copy, group=group)

    if key in ["multi_modal_inputs", "multi_modal_embeds"]:
        multi_modal_data = [item for i, item in enumerate(multi_modal_data) if i % num_repeat == 0]

    if "pixel_values_videos" in multi_modal_data[0] and num_chunk_seq > 1:
        assert multi_modal_data[0]['pixel_values_videos'].shape[0] % num_chunk_seq == 0, "The length of `pixel_values_videos` must be divisible by num_chunk_seq."
        multi_modal_data = chunk_dict_list(multi_modal_data, num_chunk_seq)

    gathered_chunks = []
    temp_chunk_group = []

    for idx, chunk in enumerate(multi_modal_data):
        if "pixel_values_videos" in chunk and num_chunk_seq > 1:
            tensor = chunk["pixel_values_videos"]
            tensor_out = [torch.empty_like(tensor) for _ in range(size)]
            dist.all_gather(tensor_out, tensor, group=group)

            meta = {k: v for k, v in chunk.items() if k != "pixel_values_videos"}
            meta_out = [None for _ in range(size)]
            dist.all_gather_object(meta_out, meta, group=group)

            chunk_output = []
            for rank_idx in range(size):
                merged = deepcopy(meta_out[rank_idx])
                merged["pixel_values_videos"] = tensor_out[rank_idx]
                chunk_output.append(merged)

            temp_chunk_group.append(chunk_output)

            if len(temp_chunk_group) == num_chunk_seq:
                gathered_chunk = []
                for rank_idx in range(size):
                    _chunk = {k: v for k, v in temp_chunk_group[0][rank_idx].items() if k != "pixel_values_videos"}
                    out = torch.empty((tensor.shape[0] * num_chunk_seq, tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
                    pos = 0
                    for d in temp_chunk_group:
                        chunk = d[rank_idx]["pixel_values_videos"]
                        chunk_len = chunk.size(0)
                        out[pos:pos + chunk_len].copy_(chunk)
                        pos += chunk_len
                    _chunk["pixel_values_videos"] = out
                    gathered_chunk.append(_chunk)
                gathered_chunks.append(gathered_chunk)
                del tensor_out
                del meta_out
                del tensor
                del meta
                temp_chunk_group = []  # Clear buffer
                torch.cuda.empty_cache()
        else:
            chunk_output = [None for _ in range(size)]
            dist.all_gather_object(chunk_output, chunk, group=group)
            gathered_chunks.append(chunk_output)

    gathered_chunks = [item for item in gathered_chunks for _ in range(num_repeat)]

    final_outputs = []
    for rank_idx in range(size):
        merged = deepcopy(base_output[rank_idx])
        merged[key] = []
        for chunk_output in gathered_chunks:
            merged[key].append(chunk_output[rank_idx])
        final_outputs.append(merged)

    return final_outputs
