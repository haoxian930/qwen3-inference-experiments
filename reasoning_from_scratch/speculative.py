"""Greedy n-gram lookup speculative decoding, extracted from the chapter 3 experiment.

Uses the textbook's Qwen3 model and dynamic KVCache; no draft model is loaded.
"""

import torch

from .ch02 import generate_text_basic_stream_cache
from .qwen3 import KVCache


def lookup_draft(history, draft_size=4, ngram_size=3):
    """Search ONLY prompt + accepted output from this request."""
    if draft_size <= 0:
        return []
    n = len(history)
    for width in range(min(ngram_size, n - 1), 0, -1):
        suffix = history[-width:]
        # Most recent prior match with at least one following token.
        for start in range(n - width - 1, -1, -1):
            if history[start:start + width] == suffix:
                return history[start + width:start + width + draft_size]
    return []


def matching_prefix(draft, predictions):
    """Only consecutive matches BEFORE the first mismatch are valid."""
    matched = 0
    for guess, expected in zip(draft, predictions):
        if guess != expected:
            break
        matched += 1
    return matched


def crop_cache(model, cache, length):
    """Roll dynamic K/V tensors and RoPE position back to the accepted prefix."""
    for layer in range(model.cfg["n_layers"]):
        kv = cache.get(layer)
        if kv is not None:
            k, v = kv
            cache.update(layer, (k[:, :, :length, :], v[:, :, :length, :]))
    model.current_pos = length


@torch.inference_mode()
def generate_lookup_speculative(
    model, token_ids, max_new_tokens, eos_token_id=None,
    draft_size=4, ngram_size=3, proposer=None,
):
    """Return generated IDs (excluding EOS) and proposal/forward-call counters.

    Supports a single nonempty prompt and the standard Qwen3 dynamic cache.
    The optional proposer(history, limit) hook supports controlled experiments.
    Block and single-token floating-point calculations can produce different
    argmaxes; benchmark token equality rather than assuming it.
    """
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise ValueError("Expected one nonempty prompt with shape (1, tokens)")
    if draft_size < 0 or ngram_size < 1:
        raise ValueError("draft_size must be >= 0 and ngram_size must be >= 1")
    stats = {"proposed": 0, "accepted": 0, "target_calls": 0}
    if max_new_tokens <= 0:
        return [], stats
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    history = token_ids[0].tolist()
    generated = []

    logits = model(token_ids, cache=cache)
    pending = int(logits[0, -1].argmax().item())
    del logits
    stats["target_calls"] += 1
    if pending == eos_token_id:
        return generated, stats
    generated.append(pending)
    history.append(pending)

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        # Reserve one output slot for the correction/bonus token.
        limit = min(draft_size, remaining - 1)
        draft = (lookup_draft(history, limit, ngram_size)
                 if proposer is None else proposer(history, limit))[:limit]
        stats["proposed"] += len(draft)
        cached_length = model.current_pos
        block = torch.tensor([pending] + draft,
                             device=token_ids.device, dtype=token_ids.dtype).unsqueeze(0)
        logits = model(block, cache=cache)
        predictions = logits[0].argmax(dim=-1).tolist()
        del logits
        stats["target_calls"] += 1

        accepted = matching_prefix(draft, predictions)
        stats["accepted"] += accepted
        # Keep pending and the accepted drafts; discard rejected K/V.
        crop_cache(model, cache, cached_length + 1 + accepted)
        additions = draft[:accepted] + [predictions[accepted]]
        for next_id in additions:
            if next_id == eos_token_id:
                return generated, stats  # EOS is excluded, like the book.
            generated.append(next_id)
            history.append(next_id)
            if len(generated) == max_new_tokens:
                return generated, stats
        pending = additions[-1]

    return generated, stats


@torch.inference_mode()
def generate_baseline(model, token_ids, max_new_tokens, eos_token_id=None):
    """Collect the textbook's cached greedy stream using the same return format."""
    if max_new_tokens <= 0:
        return [], {}
    result = []
    for token in generate_text_basic_stream_cache(
        model=model, token_ids=token_ids, max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    ):
        result.append(int(token.item()))
    return result, {}
