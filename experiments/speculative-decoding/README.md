# N-gram speculative decoding for Qwen3

This project grew out of working through Sebastian Raschka's *Build a Reasoning
Model (From Scratch)*. The question is whether reusing token sequences already in
the request can reduce target-model decoding time for math problems, and what
correctness and performance tradeoffs appear on local hardware.

## Independent contribution

The added work implements n-gram lookup proposals, target-model block verification,
matching-prefix acceptance, and KV-cache rollback. A benchmark compares baseline
and speculative decoding for Qwen3-0.6B base and reasoning checkpoints. It records
token equality as well as answer correctness, throughput, and proposal acceptance.

The Qwen3 architecture, pretrained-weight loader, tokenizer, baseline generator,
MATH-500 data loader, and answer grader come from the textbook repository. This
experiment extends that foundation; it does not train a new model or introduce a
new speculative decoding algorithm.

## Implementation

### Models and terminology

Both checkpoints use the textbook's `Qwen3Model(QWEN_CONFIG_06_B)` implementation
and its `load_model_and_tokenizer` helper. The helper downloads converted PyTorch
weights from [rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch).
Those weights originate from the official Qwen models:

| Benchmark choice | Original model | Local checkpoint |
|---|---|---|
| `base` | [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), pretrained base model | `qwen3-0.6B-base.pth` |
| `reasoning` | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), used with the book's thinking/chat template | `qwen3-0.6B-reasoning.pth` |

**Base model** identifies a checkpoint. **Baseline decoding** identifies the
ordinary cached, token-by-token greedy generation method, applied to either
checkpoint. The experiment changes the generation algorithm, not the model
weights. No new training or fine-tuning is performed.

### Code

- [Decoder](../../reasoning_from_scratch/speculative.py)
- [Benchmark CLI](../../benchmarks/speculative.py)
- [Correctness tests](../../tests/test_speculative.py)
- [Original development notebook](../../ch03/01_main-chapter-code/ch03_main.ipynb)
- [Recorded notebook results](results/notebook-summary.json)
- [Design and investigation notes](design.md)

The proposer searches the current request's prompt and accepted output for a
previous occurrence of the suffix, trying widths from three tokens down to one.
It copies up to four following tokens as a proposal. There is no second model.
The target processes the pending token and proposal in one causal forward pass.
Only the consecutively matching prefix is accepted; rejected K/V positions are
cropped, and a correction or bonus token becomes the next pending token.

## Recorded exploratory results

These numbers are transcribed from saved notebook output, not rerun measurements.
The experiment used ten MATH-500 questions, one repetition, up to 2,048 output
tokens, eager execution, bfloat16 parameters, draft size four, and n-gram size
three. CUDA was recorded; the GPU model and exact environment were not saved with
the table. The JSON labels that missing metadata explicitly.

| Model | Method | Generation seconds | Tokens/sec | Output tokens | Correct | Exact matches to baseline |
|---|---|---:|---:|---:|---:|---:|
| Base | Baseline | 43.99 | 21.73 | 956 | 2/10 | — |
| Base | Lookup speculative | 32.83 | 29.12 | 956 | 3/10 | 9/10 |
| Reasoning | Baseline | 533.85 | 22.33 | 11,919 | 7/10 | — |
| Reasoning | Lookup speculative | 274.60 | 37.27 | 10,235 | 7/10 | 0/10 |

Draft acceptance was 19.87% for the base model and 21.55% for the reasoning model.
Elapsed-time ratios were 1.34x and 1.94x respectively. **These are exploratory
ratios, not validated equivalent-output speedups.** Output sequences differ,
and the reasoning run generated fewer tokens. Ten examples cannot establish a
general accuracy change.

## Reproduce

From the repository root, with Python 3.10–3.14 and a suitable PyTorch build:

```powershell
uv sync --group dev
uv run python -m pytest tests/test_speculative.py tests/test_speculative_benchmark.py -q
uv run python -m benchmarks.speculative --models base --device cuda --examples 10 --repeats 3
```

The book's loaders download weights and MATH-500 data when missing. Base and
reasoning models run sequentially to limit memory use. Select another output path
for each run; the CLI refuses to overwrite an existing result. For a short smoke
run, use `--examples 1 --max-new-tokens 64 --repeats 1`.

The default correctness gate stops on a warmup or evaluation token mismatch and
saves diagnostics. To reproduce exploratory measurements despite divergence:

```powershell
uv run python -m benchmarks.speculative --models base reasoning --allow-mismatch --output results/exploratory.json
```

Use `--precision float32` to investigate sensitivity to precision and
`--draft-size 0` to compare the new loop without proposals. These are diagnostic
controls, not promises of token equality or speed. CPU runs are supported through
`--device cpu`.

JSON output includes software and hardware metadata, run settings, per-question
IDs and responses, generation/evaluation timing, counters, mismatch positions,
and peak CUDA allocation when available. Generation timing excludes tokenization,
text decoding, and grading; evaluation timing includes decoding and grading.
Loading and warmup are excluded. Runs alternate method order, and summaries use
the median across repeats. CUDA allocation is process allocation including model
weights, not total device VRAM use. Baseline target-call counters are unavailable.

## Status and next investigation

The notebook shows repeatable baseline output and matching output with draft
size zero, but divergence with draft size four. A saved 64-token diagnostic first
diverges at position 58. Numerical sensitivity and verification/cache errors have
not yet been ruled out on pretrained bfloat16 checkpoints.

Next steps are to inspect logits at the first divergence under the same prefix,
compare float32 and bfloat16, validate accepted cache contents, then rerun repeated
measurements on a broader set of prompts. A successful optimization must account
for correctness as well as elapsed time.

## Extraction validation

The extracted code passes 41 focused tests, including controlled cache rollback
and small random float32 Qwen3 comparisons. An end-to-end base-model smoke run on
an NVIDIA GeForce RTX 3070 Ti Laptop GPU (PyTorch 2.10.0+cu128, Python 3.10.11)
produced identical 12-token outputs for one question, with a 64-token cap.
It accepted none of its 24 proposals and was slightly slower (0.974x elapsed-time
ratio). This validates the CLI path for that case, not a general speed claim.
See the [raw smoke-run record](../../results/validation-base-final.json).

## Attribution and reading

- [Book source repository](https://github.com/rasbt/reasoning-from-scratch)
- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [Prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

The notebook also mentions DeepInfer as inspiration, but contains no specific
article URL. A precise citation is still needed before attributing a particular
technique to that source. See the repository's [Apache 2.0 license](../../LICENSE)
and retained upstream notices.
