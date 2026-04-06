"""
Basic inference script for TinyLlama-1.1B on Apple Silicon (MPS backend).

This script:
1. Loads TinyLlama-1.1B from HuggingFace
2. Runs autoregressive generation of 128 tokens
3. Measures Time to First Token (TTFT) and per-token latency
4. Verifies the setup works correctly

Key concepts explained in comments throughout.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────────────────────
# CONCEPT: Device selection
# Apple Silicon Macs have a unified memory architecture where CPU and GPU share
# the same physical RAM. PyTorch's "MPS" (Metal Performance Shaders) backend
# lets us use the GPU cores on Apple Silicon.
# ──────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
# CONCEPT: Model and Tokenizer
# A tokenizer converts human-readable text → integer IDs (tokens).
# The model takes those IDs and predicts the *next* token, one at a time.
# TinyLlama-1.1B has 1.1 billion parameters — small enough for a Mac.
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model(model_id: str, device: torch.device):
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model from {model_id} (this downloads ~2GB on first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,  # float16 halves memory usage vs float32
        low_cpu_mem_usage=True,     # stream weights to RAM instead of loading all at once
    )
    model = model.to(device)
    model.eval()  # disable dropout and batch-norm training mode
    print(f"Model loaded on {device}. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return tokenizer, model


def run_basic_inference(tokenizer, model, device, prompt: str, max_new_tokens: int = 128):
    """
    Autoregressive generation with per-token timing.

    CONCEPT: Autoregressive Decoding
    LLMs generate text one token at a time in a loop:
      Step 1: Feed the prompt tokens → model predicts token 1
      Step 2: Feed prompt + token 1 → model predicts token 2
      Step 3: Feed prompt + token 1 + token 2 → model predicts token 3
      ...and so on.
    This is called "autoregressive" because each output feeds back as input.

    CONCEPT: KV-Cache
    At each step, the model computes "Key" and "Value" matrices for every
    layer (part of the attention mechanism). Without caching, we'd recompute
    these for ALL prior tokens at every step — O(n²) cost.
    The KV-cache stores these matrices so we only compute them once per token.
    This is why step N is much faster than the prefill (first forward pass).

    CONCEPT: TTFT (Time To First Token)
    The first step processes the ENTIRE prompt at once ("prefill phase").
    This takes longer because it processes many tokens in parallel.
    TTFT = time from sending the prompt to receiving the first output token.
    """

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt length: {prompt_len} tokens")

    token_times = []  # will store latency of each generated token

    # ── Prefill Phase (produces TTFT) ──────────────────────────────────────
    # We pass the full prompt through the model once to get the first token
    # and to populate the KV-cache.
    t_prefill_start = time.perf_counter()

    with torch.no_grad():  # no gradient computation needed for inference
        # past_key_values holds the KV-cache across decoding steps
        outputs = model(input_ids=input_ids, use_cache=True)

    # Synchronize MPS (GPU ops are async; this ensures timing is accurate)
    if device.type == "mps":
        torch.mps.synchronize()

    t_prefill_end = time.perf_counter()
    ttft = t_prefill_end - t_prefill_start
    print(f"\nTime to First Token (TTFT): {ttft * 1000:.2f} ms")

    # Extract the first predicted token (greedy: pick highest-probability token)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]          # last position's logits
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy decode

    generated_ids = [next_token.item()]

    # ── Decode Phase (steady-state latency) ───────────────────────────────
    # Each subsequent step only processes ONE new token, reusing the KV-cache.
    for step in range(max_new_tokens - 1):
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,          # only the latest token
                past_key_values=past_key_values,  # reuse cached K/V from all prior tokens
                use_cache=True,
            )

        if device.type == "mps":
            torch.mps.synchronize()

        t_end = time.perf_counter()
        token_times.append(t_end - t_start)

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        # Stop if end-of-sequence token is generated
        if next_token.item() == tokenizer.eos_token_id:
            print(f"EOS token reached at step {step + 1}")
            break

    # ── Results ────────────────────────────────────────────────────────────
    import statistics
    latencies_ms = [t * 1000 for t in token_times]

    print(f"\n{'='*50}")
    print(f"Generated {len(generated_ids)} tokens")
    print(f"TTFT:                  {ttft * 1000:.2f} ms")
    print(f"Median per-token:      {statistics.median(latencies_ms):.2f} ms")
    print(f"Mean per-token:        {statistics.mean(latencies_ms):.2f} ms")
    print(f"Min per-token:         {min(latencies_ms):.2f} ms")
    print(f"Max per-token:         {max(latencies_ms):.2f} ms")
    print(f"Total generation time: {(ttft + sum(token_times)) * 1000:.2f} ms")
    print(f"{'='*50}")

    # Decode the generated token IDs back to text
    all_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=device)], dim=1)
    generated_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")

    return {
        "ttft_ms": ttft * 1000,
        "token_latencies_ms": latencies_ms,
        "generated_text": generated_text,
    }


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    tokenizer, model = load_model(MODEL_ID, device)

    PROMPT = "Explain how a computer processor works in simple terms:"

    result = run_basic_inference(
        tokenizer, model, device,
        prompt=PROMPT,
        max_new_tokens=128,
    )
