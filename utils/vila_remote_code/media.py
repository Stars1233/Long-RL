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

import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import requests
from transformers import PretrainedConfig
import imageio

# from llava.constants import MEDIA_TOKENS
# from llava.media import Image, Video
# from llava.utils import make_list
# from llava.utils.logging import logger

MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
}


class Media:
    pass


class File(Media):
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    pass


class Video(File):
    pass


def make_list(obj: Any) -> List:
    return obj if isinstance(obj, list) else [obj]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image

'''
def _load_video(video_path: str, *, num_frames: int) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            print(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]
'''
'''
def _load_video(video_path: str, *, num_frames: int) -> List[PIL.Image.Image]:
    import cv2
    from PIL import Image
    import numpy as np

    vidcap = cv2.VideoCapture(video_path)

    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        # fallback: linear scan to get count
        frame_count = 0
        while True:
            success = vidcap.grab()
            if not success:
                break
            frame_count += 1
        vidcap.release()
        vidcap = cv2.VideoCapture(video_path)

    # Compute the frame indices we want to extract
    indices = set(np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int))

    frames = []
    current_index = 0
    extracted = 0
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if current_index in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            extracted += 1
            if extracted >= num_frames:
                break
        current_index += 1

    vidcap.release()
    return frames
'''

'''
def _load_video(video_path: str, num_frames: int) -> List[PIL.Image.Image]:
    from PIL import Image

    reader = imageio.get_reader(video_path, "ffmpeg")
    nframes = reader.count_frames()
    indices = np.round(np.linspace(0, nframes - 1, num_frames)).astype(int)

    frames = []
    for i, frame in enumerate(reader):
        if i in indices:
            frames.append(Image.fromarray(frame))
        if len(frames) >= num_frames:
            break

    reader.close()
    return frames
'''

def _load_video(video_path, num_frames):
    """
    num_frames is the max number of frames the model can support.
    frame_count is the number of frames in the input video.
    max_fps is the max FPS of the model can support.
    fps is the fps of the input video.
    """

    import random
    from PIL import Image

    vidcap = cv2.VideoCapture(video_path)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print(f"Video file not found. return empty images. {video_file_name}")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames

    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    if frame_interval == 0 and frame_count <= 1:
        print(f"frame_interval is equal to 0. return empty image. {video_file_name}")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames
    # print("duration:", duration, "frames:", frame_count, "intervals:", frame_interval)

    images = []
    count = 0
    success = True
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    while success:
        # print("frame_count:", frame_count, "count:", count, "num_frames:", num_frames, "frame_interval:", frame_interval)
        if frame_count >= num_frames:
            if count in frame_indices:
                success, frame = vidcap.read()
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                if len(images) >= num_frames:
                    return images
            else:
                success = vidcap.grab()
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                count += 1
            else:
                break
    if len(images) == 0:
        raise ValueError("Did not find enough frames in the video. return empty image.")

    return images


def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    frames = _load_video(video.path, num_frames=num_frames)
    return frames


def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        print(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    media["video"].append(_extract_video(part, config))
                text += MEDIA_TOKENS["video"]
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media
