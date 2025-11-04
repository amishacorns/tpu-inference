# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Approximate KL divergence for one OpenAI-compatible vLLM server at a time.

Workflow (single-server, sequential):
- Dump: query the current server for top-K prompt_logprobs and save to disk.
- Compare: load a previous dump (model A) and compare to the current server
    (model B), computing per-position and average KL.

Approach
- For each prompt token position t, request top-K logprobs for the prompt
  from both servers (using prompt_logprobs).
- Build the union of the top-K token sets from both servers for that position.
- Convert logprobs -> probs, compute the tail mass as 1 - sum(topK_probs).
- Add an "<OTHER>" bucket to hold the tail mass for each server.
- Compute KL(P || Q) over the union ∪ {<OTHER>}.

Notes
- This is an approximation; exact KL would require the full vocabulary
  distribution. Increasing K improves accuracy. K=100..1000 is a practical range.
- Both models must share the same tokenizer/vocabulary to interpret tokens
  identically. If they do not, results are not meaningful.
- The server must support returning prompt_logprobs. For vLLM, ensure the
  server's max_logprobs is large enough (e.g., --max-logprobs 1000) and set
  logprobs_mode appropriately (e.g., raw_logprobs) when launching.

Usage
    # Phase 1: Dump current server's top-K prompt_logprobs
    python examples/quantize/compare_kl_api.py \
        --model Qwen/Qwen3-4B \
        --base-url http://localhost:8080/v1/completions \
        --k 100 --max-prompts 50 --dump dump_A.jsonl.gz

    # Phase 2: Compare a previous dump (A) against current server (B)
    python examples/quantize/compare_kl_api.py \
        --model Qwen/Qwen3-4B \
        --base-url http://localhost:8080/v1/completions \
        --k 100 --max-prompts 50 --load dump_A.jsonl.gz --save results_kl.jsonl.gz

This script issues max_tokens=1 requests (temperature=0) to fetch prompt_logprobs without meaningful generation.
"""

from __future__ import annotations

import argparse
import json
import gzip
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

import requests
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


# Verbosity toggle set by CLI
VERBOSE: bool = False


@dataclass
class Server:
    model: str
    base_url: str


def _request_prompt_logprobs(
    server: Server,
    prompt: str,
    k: int,
    timeout: float = 60.0,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Call /v1/completions with max_tokens=1, temperature=0 to retrieve prompt_logprobs.
    Returns the raw JSON response.
    """
    # Align request shape with online_perplexity_api.py
    payload = {
        "model": server.model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": int(k),
    }
    if extra:
        payload.update(extra)

    url = server.base_url.rstrip("/")
    # Support both /completions and full .../v1/completions
    if not url.endswith("/completions"):
        url = url + "/completions"

    if VERBOSE:
        print(f"POST {url} | model={server.model} k={k} max_tokens=1 temp=0 | prompt_len={len(prompt)}")
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Surface server error content to aid debugging (e.g., max_logprobs too small)
        detail = None
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        raise requests.HTTPError(f"{e} | Response: {detail}")
    return resp.json()


def _extract_prompt_topk(response: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
    """
    Extract per-position top-K logprobs from vLLM prompt_logprobs response.
    
    Expected format:
    choices[0].prompt_logprobs = [
        null,  # BOS position
        {"374": {"logprob": -6.9, "rank": 81, "decoded_token": " is"}, ...},
        {...},
        ...
    ]
    
    Returns a list (positions) of list of (token, logprob) pairs.
    """
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("No choices in response")
    c0 = choices[0]
    if VERBOSE:
        print(f"choice keys: {list(c0.keys())}")
    
    plp = c0.get("prompt_logprobs")
    if plp is None:
        if VERBOSE:
            print("prompt_logprobs missing or null. c0=", {k: type(v).__name__ for k, v in c0.items()})
        raise ValueError("Server did not return prompt_logprobs; ensure it is enabled and that max_tokens>=1 with prompt_logprobs requested")
    
    if not isinstance(plp, list):
        raise ValueError(f"prompt_logprobs must be a list; got {type(plp).__name__}")
    
    if VERBOSE:
        print(f"prompt_logprobs entries: {len(plp)}")
    
    topk_per_pos: List[List[Tuple[str, float]]] = []
    for entry in plp:
        # First position is often None (BOS)
        if entry is None:
            if VERBOSE:
                print("  - skip: None entry (likely BOS)")
            continue

        pairs: List[Tuple[str, float]] = []

        if isinstance(entry, dict):
            # vLLM common shape: {token_id: {logprob, rank, decoded_token}}
            for token_id, obj in entry.items():
                if not isinstance(obj, dict):
                    continue
                lp = obj.get("logprob")
                tok = obj.get("decoded_token") or obj.get("token")
                # Fallback to token_id if decoded string not provided
                if tok is None:
                    tok = token_id
                if lp is not None and tok is not None:
                    pairs.append((str(tok), float(lp)))

        elif isinstance(entry, list):
            # Alternate shape: list of candidate dicts
            for cand in entry:
                if not isinstance(cand, dict):
                    continue
                lp = cand.get("logprob")
                tok = cand.get("decoded_token") or cand.get("token") or cand.get("token_id")
                if lp is not None and tok is not None:
                    pairs.append((str(tok), float(lp)))

        else:
            if VERBOSE:
                print(f"  - skip: unsupported entry type {type(entry).__name__}")
            continue

        if not pairs:
            if VERBOSE:
                # For dict entries, try to show a couple keys to aid debugging
                sample_keys = list(entry.keys())[:5] if isinstance(entry, dict) else None
                print(f"  - skip: no valid candidates at this position | sample_keys={sample_keys}")
            continue

        if VERBOSE:
            print(f"  - parsed candidates: {len(pairs)} | sample: {pairs[:3]}")
        topk_per_pos.append(pairs)
    
    return topk_per_pos


def _open_text_write(path: str):
    return gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "w", encoding="utf-8")


def _save_topk_dump(path: str, prompts: List[str], all_topk: Dict[str, List[List[Tuple[str, float]]]], k: int) -> None:
    """Save per-prompt topK dump as JSONL: {prompt, topk: [[(tok, lp), ...], ...], k: int}"""
    with _open_text_write(path) as f:
        for p in prompts:
            rec = {"prompt": p, "topk": all_topk.get(p, []), "k": k}
            f.write(json.dumps(rec) + "\n")


def _load_topk_dump(path: str) -> Tuple[Dict[str, List[List[Tuple[str, float]]]], Optional[int]]:
    """
    Load a JSONL dump produced by _save_topk_dump into a dict[prompt]=topk.
    Returns (dump_dict, k_value) where k_value is the K used during dump (if recorded).
    """
    out: Dict[str, List[List[Tuple[str, float]]]] = {}
    dump_k = None
    # Support gzip-compressed dumps too
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec["prompt"]
            topk = rec["topk"]
            # Record K from dump if present
            if dump_k is None and "k" in rec:
                dump_k = rec["k"]
            # Ensure inner tuples are tuples
            fixed = []
            for pos in topk:
                fixed.append([(t, float(lp)) for t, lp in pos])
            out[p] = fixed
    return out, dump_k


def _kl_from_topk(
    topk_a: List[Tuple[str, float]],
    topk_b: List[Tuple[str, float]],
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Approximate KL(P||Q) from two top-K logprob lists by forming the union
    of tokens and adding an OTHER bucket for the tail mass.

    Notes:
    - Missing probabilities for tokens not present in a side's top-K are
      treated as zero when computing the union terms (i.e., q≈0 for A-only
      tokens). This yields an upper-bound style approximation that can be
      pessimistic when K is small.
    - The OTHER bucket uses each side's remaining mass (1 - sum(topK_probs)).
      For numerical robustness, this uses math.fsum to accumulate probs and
      clamps tails at zero.

    Returns (kl_divergence, entropy_p).
    """
    # Convert to dict and compute mass
    pa = {t: math.exp(lp) for t, lp in topk_a}
    pb = {t: math.exp(lp) for t, lp in topk_b}
    # Use math.fsum for better precision on very small probabilities
    sum_pa = math.fsum(pa.values())
    sum_pb = math.fsum(pb.values())
    tail_a = max(0.0, 1.0 - sum_pa)
    tail_b = max(0.0, 1.0 - sum_pb)

    # Union of tokens
    vocab = set(pa.keys()) | set(pb.keys())

    kl = 0.0
    entropy = 0.0
    for t in vocab:
        p = max(pa.get(t, 0.0), 0.0)
        q = max(pb.get(t, 0.0), 0.0)
        if p > 0.0:
            kl += p * (math.log(max(p, eps)) - math.log(max(q, eps)))
            entropy -= p * math.log(max(p, eps))

    # OTHER bucket
    if tail_a > 0.0:
        kl += tail_a * (math.log(max(tail_a, eps)) - math.log(max(tail_b, eps)))
        entropy -= tail_a * math.log(max(tail_a, eps))

    return float(kl), float(entropy)


def run_single_server(
    server: Server,
    prompts: Iterable[str],
    k: int,
    *,
    timeout: float = 60.0,
    extra: Optional[Dict[str, Any]] = None,
    dump_path: Optional[str] = None,
    load_path: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single-server sequential KL workflow.
    - If load_path is None and dump_path is provided: dump current server's top-K and exit.
    - If load_path is provided: load A (precomputed top-K), fetch B (current server) and compare A vs B.
      If dump_path is also provided, save B's top-K dump too.
    """
    results = []
    total_positions = 0
    sum_kl = 0.0
    sum_entropy = 0.0
    # Collected dump of current server (B)
    dump_b: Dict[str, List[List[Tuple[str, float]]]] = {}

    # Optionally load A
    load_a: Optional[Dict[str, List[List[Tuple[str, float]]]]] = None
    dump_k_a: Optional[int] = None
    if load_path:
        load_a, dump_k_a = _load_topk_dump(load_path)
        if VERBOSE:
            print(f"Loaded dump A with {len(load_a)} prompts from: {load_path}")
            if dump_k_a is not None:
                print(f"  Dump was created with k={dump_k_a}")
        
        # Validate K compatibility
        if dump_k_a is not None and k != dump_k_a:
            print(f"WARNING: Compare k={k} differs from dump k={dump_k_a}.")
            print(f"  KL approximation will use union of top-{min(k, dump_k_a)} from each side.")
            print(f"  For best accuracy, use matching K values (recommend k={dump_k_a}).")

    # Track progress stats
    processed = 0
    for prompt in tqdm(prompts, desc="KL prompts"):
        # Fetch current server (B)
        try:
            rb = _request_prompt_logprobs(server, prompt, k, timeout=timeout, extra=extra)
            if VERBOSE:
                # Print a compact view of the response shape
                ch0 = (rb.get("choices") or [{}])[0]
                print("response keys:", list(rb.keys()))
                print("choices[0] keys:", list(ch0.keys()))
            topk_b = _extract_prompt_topk(rb)
        except Exception as e:
            if VERBOSE:
                print("ERROR extracting prompt_logprobs:", e)
            results.append({"prompt": prompt, "error": f"B: {e}"})
            continue
        if VERBOSE:
            print(f"Prompt: {prompt[:60]!r}... | positions_B={len(topk_b)}")
        dump_b[prompt] = topk_b

        # If we are only dumping, no comparison needed
        if load_a is None:
            results.append({"prompt": prompt, "positions": len(topk_b), "avg_kl": None, "kl_per_pos": []})
            continue

        # Compare loaded A vs current B
        topk_a = load_a.get(prompt)
        if topk_a is None:
            results.append({"prompt": prompt, "error": "A: missing in loaded dump"})
            continue
        if VERBOSE:
            print(f"  positions_A={len(topk_a)}")

        # Align by length (use min length)
        L = min(len(topk_a), len(topk_b))
        kls = []
        entropies = []
        for t in range(L):
            kl_val, ent_val = _kl_from_topk(topk_a[t], topk_b[t])
            kls.append(kl_val)
            entropies.append(ent_val)
        avg_kl = float(sum(kls) / L) if L > 0 else None
        avg_entropy = float(sum(entropies) / L) if L > 0 else None

        results.append({
            "prompt": prompt,
            "positions": L,
            "avg_kl": avg_kl,
            "avg_entropy": avg_entropy,
            "kl_per_pos": kls,
        })
        total_positions += L
        if avg_kl is not None:
            sum_kl += sum(kls)
            sum_entropy += sum(entropies)
        
        # Print periodic progress updates
        processed += 1
        if not VERBOSE and processed % 10 == 0 and total_positions > 0:
            running_kl = sum_kl / total_positions
            running_ent = sum_entropy / total_positions
            print(f"  [{processed} prompts | {total_positions} positions] KL={running_kl:.6f} Entropy={running_ent:.4f}")

    overall = {
        "model": server.model,
        "base_url": server.base_url,
        "loaded_dump": load_path,
        "dump_k": dump_k_a,
        "compare_k": k,
        "num_items": len(results),
        "total_positions": total_positions,
        "mean_kl": (sum_kl / total_positions) if total_positions > 0 else None,
        "mean_entropy": (sum_entropy / total_positions) if total_positions > 0 else None,
        "results": results,
    }

    if save_path:
        with _open_text_write(save_path) as f:
            for item in results:
                f.write(json.dumps(item) + "\n")

    # Save current server (B) dump if requested
    if dump_path:
        _save_topk_dump(dump_path, list(prompts), dump_b, k)

    return overall


def main():
    ap = argparse.ArgumentParser(description="Single-server KL: dump or compare a loaded dump against current server using top-K prompt_logprobs")
    ap.add_argument("--model", required=False, help="Model name as served by the endpoint")
    ap.add_argument("--base-url", required=False, help="Completions endpoint base URL")
    ap.add_argument("--k", type=int, default=100, help="Top-K for logprobs")
    ap.add_argument("--dataset", type=str, default="wikitext", help="HF dataset (e.g., 'wikitext')")
    ap.add_argument("--dataset-name", type=str, default="wikitext-2-raw-v1", help="HF dataset config")
    ap.add_argument("--split", type=str, default="test", help="Dataset split")
    ap.add_argument("--ctx-len", type=int, default=1024, help="Context length per window")
    ap.add_argument("--stride", type=int, default=0, help="Stride between windows (<=0 means use ctx-len)")
    ap.add_argument("--max-eval-tokens", type=int, default=0, help="Cap total tokens (<=0 means unlimited)")
    ap.add_argument("--max-prompts", type=int, default=0, help="Max number of prompts to process (<=0 means unlimited)")
    ap.add_argument("--save", type=str, default=None, help="Save per-prompt results as JSONL")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--verbose", action="store_true", help="Print detailed debug info")
    ap.add_argument("--dump", type=str, default=None, help="Dump current server's topK to this JSONL(.gz)")
    ap.add_argument("--load", type=str, default=None, help="Load precomputed topK (A) from JSONL(.gz) and compare against current server")
    args = ap.parse_args()

    # Load wikitext dataset and create windows
    print(f"Loading dataset {args.dataset}/{args.dataset_name} split={args.split}...")
    ds_repo = args.dataset
    ds_name = args.dataset_name
    if ds_name is None and "/" in ds_repo and not ds_repo.startswith("http"):
        parts = ds_repo.split("/")
        if len(parts) == 2:
            ds_repo, ds_name = parts

    try:
        if ds_name:
            ds = load_dataset(ds_repo, ds_name, split=args.split)
        else:
            ds = load_dataset(ds_repo, split=args.split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset. Try --dataset wikitext --dataset-name wikitext-2-raw-v1.\n"
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    texts = [ex.get("text", "") for ex in ds]
    full_text = "\n\n".join([t for t in texts if t])

    # Tokenize with model's tokenizer
    print(f"Tokenizing with {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_ids = tokenizer.encode(full_text)
    print(f"Total tokens: {len(token_ids)}")

    # Create windows
    ctx_len = max(2, args.ctx_len)
    stride = args.stride if args.stride > 0 else ctx_len
    max_eval = args.max_eval_tokens if args.max_eval_tokens > 0 else float("inf")
    
    windows: List[List[int]] = []
    i = 0
    consumed = 0
    n = len(token_ids)
    while i < n - 1 and consumed < max_eval:
        end = min(i + ctx_len, n)
        win = token_ids[i:end]
        if len(win) >= 2:
            windows.append(win)
            consumed += len(win)
        if end >= n:
            break
        i += stride

    if args.max_prompts > 0:
        windows = windows[: args.max_prompts]
    print(f"Created {len(windows)} windows")

    # Decode windows to prompts
    prompts = [tokenizer.decode(w, skip_special_tokens=False, clean_up_tokenization_spaces=False) for w in windows]

    if not (args.model and args.base_url):
        raise SystemExit("--model and --base-url are required")

    global VERBOSE
    VERBOSE = bool(args.verbose)

    server = Server(model=args.model, base_url=args.base_url)

    if not args.dump and not args.load:
        raise SystemExit("Specify --dump to save topK or --load to compare a previous dump against the current server")

    overall = run_single_server(
        server,
        prompts,
        k=args.k,
        timeout=args.timeout,
        dump_path=args.dump,
        load_path=args.load,
        save_path=args.save,
    )

    print("\n=== KL Summary ===")
    print(json.dumps({k: v for k, v in overall.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()