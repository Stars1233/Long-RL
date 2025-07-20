import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from math_verify import parse, verify
from transformers import AutoModelForCausalLM
from datasets import load_dataset


QUESTION_TEMPLATE_VIDEO = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = completions
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                if "Therefore the final answer is:" in content:
                    content_match = re.search(r"<answer>Therefore the final answer is: (.*?)</answer>", content)
                else:
                    content_match = re.search(r"<answer>(.*?)</answer>", content)

                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


@torch.inference_mode()
def model_generate(model, media, prompt) -> str:
    # TODO: Please implement your model generation code here.
    # Note: The implementation should differ for VILA/Qwen-VL/LLaVA and other models.
    pass
    return response

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if "@" in args.data_path:
        data_path, data_split = args.data_path.split("@")
    else:
        data_split = "test"
    instances = load_dataset(data_path)[data_split]

    outputs = []
    if not os.path.isdir(args.output_dir):
        os.system("mkdir -p %s"%args.output_dir)

    for instance in tqdm(instances):
        video_id = instance["videos"].split("/")[-1].split(".")[0]
        output_path  = os.path.join(args.output_dir, "%s.json"%video_id)
        video_path = os.path.join(args.video_dir, instance["videos"])

        if os.path.exists(output_path):
            print("Finished processing %s." % instance["videos"])
            output = json.load(open(output_path))
            outputs.append(output)
            continue

        try:
            question = instance["problem"]
            prompt = QUESTION_TEMPLATE_VIDEO.format(question=question)

            response = model_generate(model, video_path, prompt)

            output = {"video_id": video_id, "question": question, "response": response, "answer": instance["answer"]}
            json.dump(output, open(output_path, "w"), ensure_ascii=False, indent=4)
            outputs.append(output)
        except Exception as e:
            print("Failed to process video %s."%instance["videos"], e)

    accuracies = accuracy_reward([output["response"] for output in outputs], [output["answer"] for output in outputs])
    format_accuracies = format_reward([output["response"] for output in outputs])
    # Gather and save outputs
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    metrics = {"accuracy": sum(accuracies) / len(outputs), "format_accuracy": sum(format_accuracies) / len(outputs)}
    io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
    logger.info(f"Metrics: {metrics}") 

if __name__ == "__main__":
    main()
