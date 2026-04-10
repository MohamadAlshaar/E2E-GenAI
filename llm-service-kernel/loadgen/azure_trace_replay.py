#reply runner
import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import httpx
from transformers import AutoTokenizer


@dataclass
class Result:
    ok: bool
    latency_s: float
    in_tokens: int
    out_tokens: int
    status_code: int


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def build_prompt_with_exact_tokens(tokenizer, target_tokens: int) -> Tuple[str, int]:
    """
    Create a prompt whose tokenized length is close to target_tokens under this tokenizer.
    Uses repetition of a short token sequence to control length.
    """
    if target_tokens <= 0:
        return "", 0

    base_text = "huawei"
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_ids:
        # Fallback if tokenizer returns nothing for some reason
        base_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [0]

    # Repeat token IDs until we reach target length
    reps = (target_tokens + len(base_ids) - 1) // len(base_ids)
    ids = (base_ids * reps)[:target_tokens]

    prompt = tokenizer.decode(ids, skip_special_tokens=True) #token space to text space
    actual = len(tokenizer.encode(prompt, add_special_tokens=False)) #text space to token space
    return prompt, actual


async def worker(
    wid: int,
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    service_url: str,
    temperature: float,
    stream: bool,
    served_model_name: str | None,
) -> List[Result]:
    results: List[Result] = []
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        prompt, in_toks, out_toks = item
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(out_toks),
            "temperature": float(temperature),
            "stream": bool(stream),
        }
        if served_model_name:
            body["model"] = served_model_name

        t0 = time.perf_counter()
        try:
            r = await client.post(service_url, json=body, timeout=300.0)
            _ = r.text  # force body read
            dt = time.perf_counter() - t0
            results.append(Result(ok=r.status_code == 200, latency_s=dt,
                                  in_tokens=in_toks, out_tokens=int(out_toks),
                                  status_code=r.status_code))
        except Exception:
            dt = time.perf_counter() - t0
            results.append(Result(ok=False, latency_s=dt,
                                  in_tokens=in_toks, out_tokens=int(out_toks),
                                  status_code=0))

        queue.task_done()

    return results


async def run(args) -> None:
    # Load trace
    df = pd.read_csv(args.trace_csv)
    # Azure trace columns are: TIMESTAMP, ContextTokens, GeneratedTokens
    # Be robust to capitalization/spacing variations:
    col_map = {c.lower(): c for c in df.columns}
    ctx_col = col_map.get("contexttokens")
    gen_col = col_map.get("generatedtokens")
    ts_col = col_map.get("timestamp")

    if ctx_col is None or gen_col is None:
        raise RuntimeError(f"Could not find ContextTokens/GeneratedTokens columns in {df.columns.tolist()}")

    #  sample
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    # Tokenizer (point to ocal snapshot directory)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=bool(args.local_files_only))

    # Build requests
    rows = []
    for _, row in df.iterrows():
        ctx = int(row[ctx_col])
        gen = int(row[gen_col])

        # Cap,keep some room for output
        # clamp ctx so ctx + gen <= max_model_len
        ctx = min(ctx, max(0, args.max_model_len - gen))
        gen = min(gen, args.max_out_tokens)

        prompt, actual_ctx = build_prompt_with_exact_tokens(tokenizer, ctx)
        rows.append((prompt, actual_ctx, gen))

    # If replay timestamps is requested compute arrival sleeps from TIMESTAMP
    sleeps = None
    if args.replay_timestamps and ts_col is not None:
        # Convert to numeric seconds deltas (assume ts is monotonic)
        tss = pd.to_datetime(df[ts_col], errors="coerce")
        if tss.notna().all():
            dt = tss.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0).tolist()
            sleeps = dt
        else:
            sleeps = None

    q: asyncio.Queue = asyncio.Queue()
    for i, item in enumerate(rows):
        await q.put(item)

    # Add stop sentinels
    for _ in range(args.concurrency):
        await q.put(None)

    async with httpx.AsyncClient() as client:
        tasks = []
        for wid in range(args.concurrency):
            tasks.append(asyncio.create_task(
                worker(
                    wid, q, client, args.service_url, args.temperature,
                    args.stream, args.served_model_name
                )
            ))

        t0 = time.perf_counter()

        # pacing (timestamp replay)
        if sleeps is not None:
            #  If need true pacing then what to do.........
            pass

        await q.join()
        all_results_nested = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    results = [r for sub in all_results_nested for r in sub]
    oks = [r for r in results if r.ok]
    lats = [r.latency_s for r in oks]

    total = len(results)
    ok_n = len(oks)
    fail_n = total - ok_n

    print(f"requests_total={total} ok={ok_n} fail={fail_n} wall_s={wall:.3f}")
    if lats:
        print(f"latency_s p50={percentile(lats, 0.50):.3f} p95={percentile(lats, 0.95):.3f} p99={percentile(lats, 0.99):.3f}")
        rps = ok_n / wall if wall > 0 else float("nan")
        print(f"throughput_rps={rps:.3f}")

        # token throughput estimates (requested tokens)
        in_tok = sum(r.in_tokens for r in oks)
        out_tok = sum(r.out_tokens for r in oks)
        print(f"in_tokens_total={in_tok} out_tokens_total={out_tok}")
        print(f"in_tok_per_s={in_tok / wall:.1f} out_tok_per_s={out_tok / wall:.1f} total_tok_per_s={(in_tok + out_tok) / wall:.1f}")


def main():
    import argparse
    import asyncio
    import os

    # Resolve paths relative to this file so it works no matter where to run it from.
    #  llm-service-kernel/loadgen/azure_trace_replay.py
    this_dir = os.path.dirname(os.path.abspath(__file__))          # .../llm-service-kernel/loadgen
    kernel_dir = os.path.dirname(this_dir)                         # .../llm-service-kernel

    default_trace_csv = os.path.join(
        kernel_dir, "loadgen", "workloads", "azure", "AzureLLMInferenceTrace_conv.csv"
    )

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--trace-csv",
        default=default_trace_csv,
        help="Path to Azure trace CSV (conv or code).",
    )
    ap.add_argument(
        "--service-url",
        default="http://127.0.0.1:8080/chat",
        help="Your FastAPI /chat endpoint.",
    )
    ap.add_argument(
        "--tokenizer-path",
        default="/home/mohamad/LLM-end-to-end-Service-main/Qwen2.5-0.5B-Instruct",
        help="Local model snapshot dir.",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,  # default to offline
        help="Force offline tokenizer loading.",
    )
    ap.add_argument(
        "--served-model-name",
        default=None,
        help="Optional: set 'model' field sent to service.",
    )
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Request streaming responses.",
    )
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=10000, help="Limit number of rows.")
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Must match vLLM --max-model-len.",
    )
    ap.add_argument(
        "--max-out-tokens",
        type=int,
        default=256,
        help="Clamp GeneratedTokens to this.",
    )
    ap.add_argument(
        "--replay-timestamps",
        action="store_true",
        help="not fully implemented.",
    )

    args = ap.parse_args()
    asyncio.run(run(args))



if __name__ == "__main__":
    main()
