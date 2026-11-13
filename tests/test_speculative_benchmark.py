from benchmarks.speculative import first_difference, summarize


def test_length_only_divergence():
    assert first_difference([1, 2], [1, 2, 3]) == 2
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1, 2], [1, 2]) is None


def test_throughput_is_aggregated_per_run():
    runs = [
        {"evaluation_seconds": 3, "records": [
            {"generation_seconds": 2, "ids": [1] * 10, "correct": True,
             "stats": {"proposed": 4, "accepted": 2, "target_calls": 6}}]},
        {"evaluation_seconds": 5, "records": [
            {"generation_seconds": 4, "ids": [1] * 40, "correct": False,
             "stats": {"proposed": 6, "accepted": 1, "target_calls": 8}}]},
    ]
    result = summarize(runs)
    assert result["median_tokens_per_second"] == 7.5
    assert result["acceptance_rate"] == 0.3
    assert result["generated_tokens_per_run"] == [10, 40]
    assert result["target_calls_per_run"] == [6, 8]


def test_baseline_call_count_is_unknown():
    run = {"evaluation_seconds": 1, "records": [
        {"generation_seconds": 1, "ids": [1], "correct": True, "stats": {}}]}
    assert summarize([run])["target_calls_per_run"] == [None]
