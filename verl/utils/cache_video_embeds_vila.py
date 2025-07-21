import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel

QUESTION_TEMPLATE_VIDEO = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

def _get_messages_vila(example,
                       prompt_key: str = "prompt",
                       image_key: str = "images",
                       video_key: str = "videos",
                       video_dir: str = None,):
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

def cache_video_frames(processor, dataset, video_dir, cache_dir, model_vision_encoder):
    for i, example in tqdm(enumerate(dataset)):
        video = example[video_key]
        sub_dir = os.path.dirname(video)
        cache_path = os.path.join(cache_dir, sub_dir)
        if not os.path.exists(cache_path):
            os.system("mkdir -p {}".format(cache_path))

        save_path = os.path.join(cache_dir, video.split(".")[0] + ".pt")

        if os.path.exists(save_path):
            print("Processed video {}".format(video))
            continue

        messages, prompt = _get_messages_vila(example, "problem", "images", video_key, video_dir)
        messages = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = processor(text=[messages], return_tensors="pt")
        with torch.no_grad():
            inputs_embeds = model_vision_encoder._embed_media_tokens(model_inputs['media'], {"video": {}})

        video_embed = inputs_embeds['video'][0]
        torch.save(video_embed, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nframes", type=int, default=256)
    parser.add_argument("--video_dir", type=str, default="./path/to/longvila_r1_data")
    parser.add_argument("--cache_dir", type=str, default="./path/to/longvila_r1_cache_vila_embed")
    parser.add_argument("--model_path", type=str, default="./path/to/model")
    parser.add_argument("--video_key", type=str, default="videos")

    args = parser.parse_args()
    dataset = load_dataset(repo_name, split="train")
    video_dir = args.video_dir
    cache_dir = args.cache_dir
    model_path = args.model_path
    os.system("mkdir -p {}".format(cache_dir))
    video_key = args.video_key
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.config.num_video_frames = args.nframes
    processor.config.fps = 2

    model_vision_encoder = AutoModel.from_pretrained(model_path,
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     llm_only_need_embed=True)
    model_vision_encoder.mm_projector = model_vision_encoder.mm_projector.cuda()

    cache_video_frames(processor, dataset, video_dir, cache_dir, model_vision_encoder)
