import argparse
import os
import json
import math
import time
from tqdm import tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt
import tempfile

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
 
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

def expand2square(pil_img, background_color=(0,0,0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def preprocess_images_to_paths_for_qwen(
    image_paths,
):
    out_paths = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")

        fill = tuple(int(x * 255) for x in (0.48145466, 0.4578275, 0.40821073))
        img = expand2square(img, background_color=fill)

        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp, format="PNG")
        out_paths.append(tmp)

    return out_paths

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
        fp16=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    print("[INFO] Model loaded.\n")

    with open(args.question_file, "r") as f:
        questions = [json.loads(line) for line in f]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        questions = questions[:args.max_samples]

    questions = questions[:5]

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    out = open(args.answers_file, "w")

    print("Starting inference…\n")

    for idx, item in enumerate(tqdm(questions, desc="Processing")):

        img_path = get_image_path(item["image_id"], args.image_folder)
        if img_path is None:
            continue

        question = item.get("query_prompt", "")
        tmp_paths = preprocess_images_to_paths_for_qwen([img_path])

        query = tokenizer.from_list_format([
            {"image": tmp_paths[0]},
            {"text": question},
        ])
        

        chat_kwargs = {
            "max_new_tokens": args.max_new_tokens,
        }

        if args.use_dtc == "True":
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
            "relation_type": item.get("relation_type", None)
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

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_dtc", type=str, default="False")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    eval_model(args)


if __name__ == "__main__":
    main()
