# Design and correctness investigation

## Cache invariant

At the start of each verification iteration, the cache represents all emitted
tokens except the most recent pending token. The pending token is already in the
output history. Forwarding `[pending] + draft` returns predictions for the draft
positions plus one correction/bonus position. For `a` matching proposals, keep
`cached_length + 1 + a` cache positions, append those proposals and prediction `a`,
and leave that last prediction pending. EOS is excluded from the returned IDs,
matching the textbook baseline.

Reserve one output slot for the correction/bonus token so verification never
needs to exceed the requested output budget. Prefix matching stops at the first
mismatch, even if a later proposed token matches its prediction. The module uses
the standard dynamic `qwen3.KVCache`; it is not a drop-in implementation for the
preallocated optimized cache or batched generation. It uses greedy argmax, not
distribution-preserving speculative sampling.

## What the saved diagnostics establish

- Baseline versus repeated baseline: identical for the saved 64-token check.
- Baseline versus draft size zero: identical for that check.
- Baseline versus draft size four: first divergence at token 58 (zero based).
- That check proposed 113 tokens, accepted six, and used 58 target calls.
- The larger saved comparison matched 9/10 base outputs and 0/10 reasoning outputs.

This isolates a useful trigger: enabling multi-token verification. It does not
identify the cause. A different matrix shape can change floating-point results,
but an indexing, masking, or cache bug remains an alternative explanation.

## Validation strategy

Tests use an exact integer-driven model whose predictions depend on the complete
cached prefix. Controlled proposals exercise complete acceptance, partial
acceptance followed by rejection, and complete rejection. They compare complete
output sequences against the book's baseline, including EOS and token budgets.
Separate tests compare a small random float32 Qwen3 model across draft sizes and
seeds. No weights or external services are needed for these tests.

Passing these tests checks algorithm behavior in controlled settings; it does
not resolve the saved pretrained bfloat16 divergence. The benchmark enforces
token equality by default and writes diagnostic records before stopping.

For the pretrained investigation, capture logits and top-two margins at the first
divergence while replaying the same prefix, compare sequential and block execution
in float32/bfloat16, and compare every accepted cache layer against sequential
execution. Do this before reporting equivalent-output speedups. Changed output
lengths and answer grades must remain visible in any exploratory results.
