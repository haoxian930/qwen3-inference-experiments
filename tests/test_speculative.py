"""Correctness checks require no pretrained weights, tokenizer, or network."""

import pytest
import torch

from reasoning_from_scratch.qwen3 import Qwen3Model
from reasoning_from_scratch.speculative import generate_baseline, generate_lookup_speculative


def next_id(history):
    return (sum(history) + len(history)) % 7


class ExactModel(torch.nn.Module):
    """Exact integer next-token rule; cache content affects every prediction."""

    def __init__(self):
        super().__init__()
        self.cfg = {"n_layers": 1}
        self.current_pos = 0

    def reset_kv_cache(self):
        self.current_pos = 0

    def forward(self, ids, cache):
        old = cache.get(0)
        history = [] if old is None else old[0].flatten().long().tolist()
        assert len(history) == self.current_pos
        logits = torch.full((1, ids.shape[1], 8), -100.0)
        for i, token in enumerate(ids[0].tolist()):
            history.append(token)
            logits[0, i, next_id(history)] = 100
        kv = torch.tensor(history).reshape(1, 1, -1, 1)
        cache.update(0, (kv, kv.clone()))
        self.current_pos = len(history)
        return logits


def oracle(history, limit):
    history = list(history)
    result = []
    for _ in range(limit):
        token = next_id(history)
        result.append(token)
        history.append(token)
    return result


def bad_after_one(history, limit):
    draft = oracle(history, limit)
    if len(draft) > 1:
        draft[1] = (draft[1] + 1) % 7
    return draft


@pytest.mark.parametrize("proposer", [None, oracle, bad_after_one, lambda history, limit: [7] * limit])
@pytest.mark.parametrize("budget", [0, 1, 2, 7, 23])
def test_cache_rollback_and_budget(proposer, budget):
    prompt = torch.tensor([[1, 2, 1, 2]])
    expected, _ = generate_baseline(ExactModel(), prompt, budget)
    actual, _ = generate_lookup_speculative(ExactModel(), prompt, budget, proposer=proposer)
    assert actual == expected
    assert len(actual) == budget


@pytest.mark.parametrize("draft_size", [0, 1, 4])
@pytest.mark.parametrize("eos", [0, 2, 5, 7])
def test_eos_excluded(draft_size, eos):
    prompt = torch.tensor([[1, 2, 1, 2]])
    expected, _ = generate_baseline(ExactModel(), prompt, 30, eos)
    actual, _ = generate_lookup_speculative(ExactModel(), prompt, 30, eos,
                                           draft_size=draft_size, proposer=oracle)
    assert actual == expected
    assert eos not in actual


def test_good_proposals_reduce_target_calls():
    prompt = torch.tensor([[1, 2]])
    actual, stats = generate_lookup_speculative(ExactModel(), prompt, 20, proposer=oracle)
    assert len(actual) == 20
    assert stats["accepted"] == stats["proposed"] > 0
    assert stats["target_calls"] < len(actual)


@pytest.mark.parametrize("seed", [1, 42, 123])
def test_real_qwen_float32(seed):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=32, context_length=96, emb_dim=16, n_heads=2,
               n_layers=2, hidden_dim=32, head_dim=8, qk_norm=True,
               n_kv_groups=1, rope_base=10000.0, dtype=torch.float32)
    model = Qwen3Model(cfg).eval()
    prompt = torch.tensor([[1, 2, 3, 1, 2, 3]])
    expected, _ = generate_baseline(model, prompt, 24)
    for size in (0, 1, 4):
        actual, _ = generate_lookup_speculative(model, prompt, 24, draft_size=size)
        assert actual == expected


@pytest.mark.parametrize("prompt", [torch.empty((1, 0), dtype=torch.long), torch.ones((2, 2), dtype=torch.long)])
def test_invalid_prompt(prompt):
    with pytest.raises(ValueError, match="nonempty prompt"):
        generate_lookup_speculative(ExactModel(), prompt, 5)
