"""
Rewriter Inference (optimized for 24GB GPU)
- Higher throughput batching with OOM fallback
- Uses inference_mode + autocast
- Avoids torch.cuda.empty_cache() in the hot loop
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
transformers.logging.set_verbosity_error()

class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Rewriter Inference.")
    parser.add_argument("--prompt_path", required=True, help="Path to prompt json")
    parser.add_argument("--model_path", default="/teamspace/studios/this_studio/travail/models/qwen-7b", help="Local Qwen path")
    parser.add_argument("--output_path", default="/teamspace/studios/this_studio/travail/CodeFuse-CGM/rewriter/test_rewriter_output.json")
    parser.add_argument("--batch_size", type=int, default=16, help="Start batch size (will auto fallback on OOM)")
    parser.add_argument("--max_input_tokens", type=int, default=2048, help="Max tokens for prompt inputs")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()

def build_chat_text(tokenizer, prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def run_inference(prompts, model, tokenizer, batch_size, max_input_tokens, max_new_tokens):
    gen_cfg = GenerationConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    responses = []
    i = 0
    pbar = tqdm(total=len(prompts), desc="Rewriter", unit="prompt")

    while i < len(prompts):
        bs = min(batch_size, len(prompts) - i)
        batch = prompts[i:i+bs]

        try:
            text_batch = [build_chat_text(tokenizer, p) for p in batch]

            model_inputs = tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                # autocast speeds up on GPU
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    generated_ids = model.generate(**model_inputs, generation_config=gen_cfg)

            # remove prompt tokens from outputs
            trimmed = []
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                trimmed.append(output_ids[len(input_ids):])

            out = tokenizer.batch_decode(trimmed, skip_special_tokens=True)
            responses.extend(out)

            i += bs
            pbar.update(bs)

        except torch.cuda.OutOfMemoryError:
            # fallback: reduce batch size and retry
            torch.cuda.empty_cache()
            if batch_size == 1:
                # cannot reduce further; mark as error
                responses.extend(["error"] * bs)
                i += bs
                pbar.update(bs)
            else:
                batch_size = max(1, batch_size // 2)
                print(f"\n[OOM] Reducing batch_size to {batch_size} and retrying...", flush=True)

        except Exception as e:
            # other errors: mark this batch as error and continue
            responses.extend(["error"] * bs)
            i += bs
            pbar.update(bs)

    pbar.close()
    return responses

def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    df = pd.read_json(args.prompt_path)

    extractor_prompts = df["extractor_prompt"].tolist()
    inferer_prompts = df["inferer_prompt"].tolist()
    instance_num = len(extractor_prompts)

    # Keep ordering consistent: first extractor then inferer (or vice versa). Choose one and stay consistent.
    all_prompts = extractor_prompts + inferer_prompts

    responses = run_inference(
        all_prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    df["rewriter_extractor"] = responses[:instance_num]
    df["rewriter_inferer"] = responses[instance_num:instance_num*2]

    df.to_json(args.output_path, index=False)
    print("Saved:", args.output_path)

if __name__ == "__main__":
    main()