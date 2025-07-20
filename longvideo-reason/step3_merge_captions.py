import os
import json
from argparse import ArgumentParser


def collect_video_files(folder):
    video_files = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            parts = file_name.split("_")
            video_name = "_".join(parts[:-1])
            if video_name not in video_files:
                video_files[video_name] = []
            video_files[video_name].append(file_name)
    return video_files


def merge_captions(video_files, input_folder):
    merged_data = []
    for video_name, file_list in video_files.items():
        output_dict = {"id": video_name}
        merged_captions = []
        for file_name in sorted(file_list):
            file_path = os.path.join(input_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, caption in data.items():
                    index = int(key.split("_")[-1])
                    start_time = index * 10
                    end_time = (index + 1) * 10
                    merged_captions.append({"start_time": start_time, "end_time": end_time, "caption": caption.replace('\n', ' ')})
        output_dict["captions"] = merged_captions
        merged_data.append(output_dict)
    return merged_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="path/to/short_clip_captions")
    parser.add_argument("--output_file", type=str, default="./merged_captions.json")
    args = parser.parse_args()

    video_files = collect_video_files(args.input_dir)
    merged_data = merge_captions(video_files, args.input_dir)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"Finished！Results merged into {args.output_file}.")
