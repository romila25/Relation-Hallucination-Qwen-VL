import argparse
import os
import json
import math
import time
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_image_path(image_id, root):
    iid = str(image_id).replace(".jpg", "")
    p1 = os.path.join(root, "VG_100K", f"{iid}.jpg")
    p2 = os.path.join(root, "VG_100K_2", f"{iid}.jpg")
    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2
    print(f"[WARN] Missing image {iid}.jpg")
    return None


def split_list(lst, n):
    if n <= 0:
        return [lst]
    chunk = math.ceil(len(lst) / n)
    return [lst[i:i+chunk] for i in range(0, len(lst), chunk)]


def get_chunk(lst, n_chunks, idx):
    chunks = split_list(lst, n_chunks)
    return chunks[idx] if 0 <= idx < len(chunks) else []


# ---------------------------------------------------------
# Main inference logic
# ---------------------------------------------------------

@torch.inference_mode()
def eval_model(args):
    print("\n==============================")
    print("  Qwen-VL-Chat (FP16, GPU)")
    print("==============================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model_path = os.path.expanduser(args.model_path)

    # Load processor
    print("[INFO] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Load model fully on GPU
    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    # Load questions
    print("[INFO] Reading questions:", args.question_file)
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(line) for line in f]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.max_samples:
        questions = questions[:args.max_samples]

    # Output file
    answers_path = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    out = open(answers_path, "w")

    # Loop
    for idx, item in enumerate(tqdm(questions)):

        # Get image
        img_path = get_image_path(item["image_id"], args.image_folder)
        if not img_path:
            continue

        image = Image.open(img_path).convert("RGB")
        raw_question = item.get("query_prompt", "")

        # Build the Qwen-VL-Chat style message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": raw_question + "\nOnly answer yes or no."},
                ],
            }
        ]

        # Build input_ids from chat template
        text_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Process image for the model
        vision_inputs = processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].to(device)

        # Merge inputs
        inputs = {
            "input_ids": text_inputs,
            "images": vision_inputs,
        }

        # Generation settings
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            if args.top_p:
                gen_kwargs["top_p"] = args.top_p

        start = time.time()

        output_ids = model.generate(
            **inputs,
            **gen_kwargs
        )

        elapsed = time.time() - start

        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        record = {
            "image_id": item.get("image_id"),
            "query_prompt": raw_question,
            "response": response,
            "label": item.get("label"),
            "mllm_name": "Qwen-VL-Chat-FP16",
            "inference_time": elapsed,
        }

        out.write(json.dumps(record) + "\n")
        out.flush()

    out.close()
    print("\n🎉 DONE — Results saved to:", answers_path)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()
    eval_model(args)


if __name__ == "__main__":
    main()
