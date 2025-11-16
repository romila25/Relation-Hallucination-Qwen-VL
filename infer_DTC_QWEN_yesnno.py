import argparse
import os
import json
import math
import time
from tqdm import tqdm
from PIL import Image

from DTC import DTC_function
from reefknot_qwen import ReefknotQwen

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


DTC_function()
# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_image_path(image_id, root):
    """Return full path to image in VG_100K or VG_100K_2."""
    iid = str(image_id).replace(".jpg", "")
    p1 = os.path.join(root, "VG_100K", f"{iid}.jpg")
    p2 = os.path.join(root, "VG_100K_2", f"{iid}.jpg")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    print(f"[WARN] Missing image {iid}.jpg")
    return None


def split_list(lst, n):
    if n <= 0:
        return [lst]
    chunk = math.ceil(len(lst) / n)
    return [lst[i:i + chunk] for i in range(0, len(lst), chunk)]


def get_chunk(lst, n_chunks, idx):
    chunks = split_list(lst, n_chunks)
    return chunks[idx] if 0 <= idx < len(chunks) else []


# ---------------------------------------------------------
# Main inference
# ---------------------------------------------------------

@torch.inference_mode()
def eval_model(args):
    print("\n==============================")
    print("  Qwen-VL-7B (FP16, full GPU)")
    print("==============================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model_path = os.path.expanduser(args.model_path)
    print(f"[INFO] Loading model from: {model_path}")
    
    # ------------------------------
    # Load model fully on GPU (no offload)
    # ------------------------------
    print("[INFO] Loading model in FP16 on GPU (no offload)...")

    model = ReefknotQwen(model_path, dtype=torch.float16)

    processor = model.processor
    
    
    # ------------------------------
    # Load processor
    # ------------------------------
    print("[INFO] Loading processor...")
    
    p = next(model.parameters())
    print(f"[INFO] Model param: dtype={p.dtype}, device={p.device}\n")

    # ------------------------------
    # Load questions
    # ------------------------------
    print("[INFO] Reading questions:", args.question_file)
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(line) for line in f]

    print(f"[INFO] Total samples in file: {len(questions)}")

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"[INFO] Using chunk {args.chunk_idx+1}/{args.num_chunks} -> {len(questions)} samples")

    if args.max_samples:
        questions = questions[:args.max_samples]
        print(f"[INFO] Limiting to first {args.max_samples} samples\n")

    # ------------------------------
    # Prepare output
    # ------------------------------
    answers_path = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    out = open(answers_path, "w")

    print("[INFO] Starting inference...\n")

    # ------------------------------
    # Inference loop
    # ------------------------------
    for idx, item in enumerate(tqdm(questions, desc="Processing")):

        img_path = get_image_path(item["image_id"], args.image_folder)
        if not img_path:
            continue

        image = Image.open(img_path).convert("RGB")
        # prompt = item.get("query_prompt", "")
        raw_question = item.get("query_prompt", "")
        raw_label = item.get("label", "")

        prompt = (
            "<|im_start|>user\n"
            + raw_question
            + "\nOnly answer yes or no.\n"
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )

        # Build inputs on CPU, then move tensors to GPU
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generation params
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
        }
        if args.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = args.temperature
            if args.top_p:
                gen_kwargs["top_p"] = args.top_p
        else:
            gen_kwargs["do_sample"] = False

        start = time.time()

        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

        elapsed = time.time() - start

        response = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()

        if idx < 2:
            print("\n========== SAMPLE OUTPUT ==========")
            print("Image:", img_path)
            print("Prompt:", prompt)
            print("Response:", response)
            print('Label:', raw_label)
            print("Time:", f"{elapsed:.2f}s")
            print("===================================\n")

        record = {
            "image_id": item.get("image_id"),
            "query_prompt": prompt,
            "response": response,
            "label": item.get("label"),
            "mllm_name": "Qwen-VL-7B-FP16-GPU",
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

    parser.add_argument("--model-path", type=str, default="/kaggle/working/Qwen-VL-7B")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    
    parser.add_argument("--apha", type=float, default=0.1)
    parser.add_argument("--layer", type=int, default=38)
    parser.add_argument("--threshold", type=float, default=0.9)

    args = parser.parse_args()
    eval_model(args)


if __name__ == "__main__":
    main()