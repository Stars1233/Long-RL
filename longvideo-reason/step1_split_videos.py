import os
import subprocess
from tqdm import tqdm
from moviepy import VideoFileClip
from argparse import ArgumentParser

def split_video_into_clips(input_path, output_dir, clip_duration=10):
    video_name = os.path.splitext(os.path.basename(input_path))[0]

    print("video_name", video_name)
    try:
        clip = VideoFileClip(input_path)
        duration = int(clip.duration)
        clip.reader.close()
        if clip.audio is not None:
            clip.audio.reader.close()
    except Exception as e:
        print(f"Failed to load video: {input_path} with error: {e}")
        return

    for start in tqdm(range(0, duration, clip_duration)):
        out_file = os.path.join(output_dir, f"{video_name}_{start:04d}.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(clip_duration),
            "-i", input_path,
            "-c", "copy",
            out_file
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Finished splitting: {input_path}")

def batch_split_videos(input_dir, output_dir, clip_duration=10):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
            input_path = os.path.join(input_dir, file_name)
            split_video_into_clips(input_path, output_dir, clip_duration)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="path/to/videos")
    parser.add_argument("--output_dir", type=str, default="path/to/output_video_clips")
    parser.add_argument("--clip_duration", type=int, default=10, help="duration of each clip")
    args = parser.parse_args()

    batch_split_videos(args.input_dir, args.output_dir, args.clip_duration)
