import argparse
import os
import json
import math
import time
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from DTC import DTC_function

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
    print(" Qwen-VL-Chat ")
    print("==============================\n")

    model_path = os.path.expanduser(args.model_path)
    print(f"[INFO] Loading model from: {model_path}")

    # ----- Load tokenizer -----
    print("[INFO] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # ----- Load model -----
    print("[INFO] Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    print("[INFO] Model loaded.\n")

    torch.set_autocast_gpu_dtype(torch.float16)
    transformers.utils.import_utils._pt_autocast_enabled = False
    
    with open(args.question_file, "r") as f:
        questions = [json.loads(line) for line in f]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        questions = questions[:args.max_samples]

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    out = open(args.answers_file, "w")

    print("Starting inference…\n")

    for idx, item in enumerate(tqdm(questions, desc="Processing")):

        img_path = get_image_path(item["image_id"], args.image_folder)
        if img_path is None:
            continue

        question = item.get("query_prompt", "")

        print(item["image_id"])
        
        query = tokenizer.from_list_format([
            {"image": img_path},
            {"text": question},
        ])

        chat_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "model_path": model_path, 
            "apha": args.apha,
            "layer" : args.layer,
            "threshold" : args.threshold
        }

        if args.temperature is not None and args.temperature > 0:
            chat_kwargs["do_sample"] = True
            chat_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                chat_kwargs["top_p"] = args.top_p
        else:
            chat_kwargs["do_sample"] = False

        start = time.time()

        try:
            response, _ = model.chat(
                tokenizer,
                query=query,
                history=None,
                **chat_kwargs,
            )
        except TypeError:
            response, _ = model.chat(
                tokenizer,
                query=query,
                history=None,
            )

        elapsed = time.time() - start

        record = {
            "image_id": item["image_id"],
            "query_prompt": question,
            "response": response,
            "label": item.get("label", None),
            "mllm_name": "Qwen-VL-Chat",
            "inference_time": elapsed,
        }

        out.write(json.dumps(record) + "\n")
        out.flush()

        if idx < 2:
            print("\n--- SAMPLE OUTPUT ---")
            print("Image:", img_path)
            print("Q:", question)
            print("A:", response)
            print("Time:", f"{elapsed:.2f}s")
            print("---------------------\n")

    out.close()
    print("\nDone! Saved results to:", args.answers_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_dtc", type=bool, default=False)
    
    parser.add_argument("--apha", type=float, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    
    args = parser.parse_args()
    
    if args.use_dtc == True:
        DTC_function()
        
    eval_model(args)


if __name__ == "__main__":
    main()
