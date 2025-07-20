import os
import json
import openai
from tqdm import tqdm
from argparse import ArgumentParser


def generate_captions(caption_dict):

    output = ""
    if len(caption_dict) == 0:
        return output
    for item in caption_dict:
        caption_text = item['caption'][0].lower() +item['caption'][1:] if item['caption'] else ""
        output += f"From {item['start_time']} to {item['end_time']}, {caption_text}\n"
    return output

def main(
    client,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    captions: str = "path/to/merged_captions.json",
    output_folder: str = "./output_folder",
) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    merged_captions = json.load(open(captions))

    for i, item in tqdm(enumerate(merged_captions)):
        video_name = item['id']
        item_file = f'{output_folder}/{video_name}.json'
        if os.path.exists(item_file):
            print(f"{item_file} has already been processed.")
            continue

        captions = generate_captions(item["captions"])
        if len(captions) == 0:
            continue

        output_dict = {"video_name": video_name, "captions": captions}
        prompt = (
            f"Based on the following captions for a video, generate a challenging multiple-choice question that requires **multiple reasoning steps** and deep understanding to answer. "
            f"The question should involve as many logical steps as possible, ensuring that the answer cannot be deduced without careful analysis of the captions. "
            f"Provide the question with four options (A, B, C, D), clearly indicating the correct answer, and include detailed reasoning with timestamps.\n\n"
            f"The question should be related to Goal and Intention Reasoning.\n" # NOTE: Change here to other descriptions to align with your own data.
            f"Captions:\n{captions}\n\nOutput format:\n"
            f"QUESTION: <Your question>\n"
            f"OPTIONS:\n"
            f"A. <Option A>\n"
            f"B. <Option B>\n"
            f"C. <Option C>\n"
            f"D. <Option D>\n"
            f"ANSWER: <Correct answer (e.g., A, B, C, or D)>\n"
            f"REASONS:\n"
            f"##### From [start to end]:\n"
            f"- <Reason 1>\n"
            f"- <Reason 2>\n"
            f"- <Reason 3>\n"
            f"##### From [start to end]:\n"
            f"- <Reason 4>\n"
            f"- <Reason 5>\n"
            f"##### (Add as many steps as needed, grouping reasons under shared timestamps where applicable)"
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        completion = response.choices[0].message.content.strip()

        output_dict["reasoning"] = completion
        with open(item_file, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
    parser = ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--captions", type=str, default="path/to/merged_captions.json")
    parser.add_argument("--output_folder", type=str, default="./output_folder")
    args = parser.parse_args()
    main(client, args.max_new_tokens, args.temperature, args.captions, args.output_folder)
