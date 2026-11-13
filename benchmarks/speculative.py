"""Compare cached greedy decoding with n-gram lookup speculative decoding.

Run from the repository root: python -m benchmarks.speculative --help
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import torch

from reasoning_from_scratch.ch03 import (
    extract_final_candidate, grade_answer, load_math500_test,
    load_model_and_tokenizer, render_prompt,
)
from reasoning_from_scratch.speculative import generate_baseline, generate_lookup_speculative


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def summarize(runs):
    """Aggregate complete runs; never divide a median token count by median time."""
    totals = [sum(row["generation_seconds"] for row in run["records"]) for run in runs]
    counts = [sum(len(row["ids"]) for row in run["records"]) for run in runs]
    proposed = sum(row["stats"].get("proposed", 0) for run in runs for row in run["records"])
    accepted = sum(row["stats"].get("accepted", 0) for run in runs for row in run["records"])
    return {
        "median_generation_seconds": statistics.median(totals),
        "median_tokens_per_second": statistics.median(n / t for n, t in zip(counts, totals)),
        "generated_tokens_per_run": counts,
        "correct_per_run": [sum(row["correct"] for row in run["records"]) for run in runs],
        "median_evaluation_seconds": statistics.median(run["evaluation_seconds"] for run in runs),
        "accepted": accepted, "proposed": proposed,
        "acceptance_rate": accepted / proposed if proposed else None,
        "target_calls_per_run": [sum(row["stats"]["target_calls"] for row in run["records"])
                                 if all("target_calls" in row["stats"] for row in run["records"])
                                 else None for run in runs],
    }


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["base", "reasoning"], default=["base", "reasoning"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--draft-size", type=int, default=4)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--model-dir", default="qwen3")
    parser.add_argument("--data", default="ch03/01_main-chapter-code/math500_test.json")
    parser.add_argument("--output", type=Path, default=Path("results/speculative-run.json"))
    parser.add_argument("--allow-mismatch", action="store_true", help="Continue exploratory timing after token divergence")
    args = parser.parse_args()
    if min(args.examples, args.max_new_tokens, args.repeats, args.ngram_size) < 1 or args.draft_size < 0:
        parser.error("Counts must be positive; draft size may be zero")
    if args.output.exists():
        parser.error("Output already exists; choose another path to preserve the previous run")
    return args


def main():
    # The upstream weight loader prints Unicode status marks on legacy consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    examples = load_math500_test(args.data)[:args.examples]
    if not examples:
        raise ValueError("The dataset is empty")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {**vars(args), "output": str(args.output)},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "execution": "eager", "seed": args.seed,
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "models": {},
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for kind in args.models:
        print(f"Loading {kind}...", flush=True)
        model, tokenizer = load_model_and_tokenizer(kind, device, use_compile=False, local_dir=args.model_dir)
        dtype = getattr(torch, args.precision)
        model.to(dtype=dtype).eval()
        # The book uses cfg['dtype'] when projecting final logits.
        model.cfg = {**model.cfg, "dtype": dtype}
        inputs = [torch.tensor(tokenizer.encode(render_prompt(row["problem"])), device=device).unsqueeze(0)
                  for row in examples]
        if any(ids.shape[1] + args.max_new_tokens > model.cfg["context_length"] for ids in inputs):
            raise ValueError("Prompt plus requested output exceeds context length")

        def generate(method, ids, limit, target=model, eos=tokenizer.eos_token_id):
            if method == "baseline":
                return generate_baseline(target, ids, limit, eos)
            return generate_lookup_speculative(target, ids, limit, eos,
                                               draft_size=args.draft_size, ngram_size=args.ngram_size)

        warm_limit = min(64, args.max_new_tokens)
        warm_base, _ = generate("baseline", inputs[0], warm_limit)
        warm_spec, _ = generate("speculative", inputs[0], warm_limit)
        synchronize(device)
        result = {"warmup_identical": warm_base == warm_spec,
                  "warmup_first_difference": first_difference(warm_base, warm_spec),
                  "warmup_ids": {"baseline": warm_base, "speculative": warm_spec},
                  "runs": {"baseline": [], "speculative": []}}
        report["models"][kind] = result
        if warm_base != warm_spec and not args.allow_mismatch:
            save()
            raise SystemExit(f"{kind}: warmup tokens diverged; diagnostics saved to {args.output}. "
                             "Use --allow-mismatch only for exploratory measurements.")

        for repeat in range(args.repeats):
            order = ("baseline", "speculative") if repeat % 2 == 0 else ("speculative", "baseline")
            for method in order:
                print(f"{kind}: {method}, run {repeat + 1}/{args.repeats}", flush=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                synchronize(device)
                evaluation_start = time.perf_counter()
                records = []
                for example, input_ids in zip(examples, inputs):
                    synchronize(device)
                    start = time.perf_counter()
                    ids, stats = generate(method, input_ids, args.max_new_tokens)
                    synchronize(device)
                    seconds = time.perf_counter() - start
                    response = tokenizer.decode(ids)
                    answer = extract_final_candidate(response)
                    records.append({"problem": example["problem"], "ground_truth": example["answer"],
                                    "ids": ids, "response": response, "answer": answer,
                                    "correct": bool(grade_answer(answer, example["answer"])),
                                    "generation_seconds": seconds, "stats": stats})
                synchronize(device)
                result["runs"][method].append({
                    "records": records, "evaluation_seconds": time.perf_counter() - evaluation_start,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else None,
                })
            base = result["runs"]["baseline"][-1]["records"]
            spec = result["runs"]["speculative"][-1]["records"]
            differences = [{"question": i + 1, "first_token_difference": first_difference(a["ids"], b["ids"])}
                           for i, (a, b) in enumerate(zip(base, spec)) if a["ids"] != b["ids"]]
            result.setdefault("differences_per_run", []).append(differences)
            save()
            if differences and not args.allow_mismatch:
                raise SystemExit(f"{kind}: {len(differences)} outputs diverged; diagnostics saved to {args.output}")

        result["summary"] = {method: summarize(runs) for method, runs in result["runs"].items()}
        result["elapsed_time_ratio"] = (result["summary"]["baseline"]["median_generation_seconds"] /
                                        result["summary"]["speculative"]["median_generation_seconds"])
        result["all_outputs_identical"] = not any(result["differences_per_run"])
        result["interpretation"] = "matched outputs" if result["all_outputs_identical"] else "exploratory: outputs differ"
        print(f"{kind}: {result['elapsed_time_ratio']:.3f}x elapsed-time ratio; {result['interpretation']}")
        save()
        del model, tokenizer, inputs, generate
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Raw records and metadata: {args.output}")


if __name__ == "__main__":
    main()
